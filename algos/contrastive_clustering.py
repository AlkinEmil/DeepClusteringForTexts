import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.cohiclust_utils import Banking1NNPair

class VanillaMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        inp_dim = cfg.model.inp_dim
        out_dim = cfg.model.out_dim
        linear_dims_list = cfg.model.linear_dims_list
        drop_prob = cfg.model.drop_prob
        assert len(linear_dims_list) > 1, "Need at least 2 linear layers"
        
        self.inp_layer = nn.Sequential(
            nn.Linear(inp_dim, linear_dims_list[0]), 
            nn.ReLU(),
            nn.BatchNorm1d(linear_dims_list[0]),
            nn.Dropout(drop_prob)
        )
        
        self.main_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(linear_dims_list[i], linear_dims_list[i + 1]), 
                nn.ReLU(),
                nn.BatchNorm1d(linear_dims_list[i + 1]),
                nn.Dropout(drop_prob)
            )
            for i in range(len(linear_dims_list) - 1)
        ])
        
        self.out_layer = nn.Linear(linear_dims_list[-1], out_dim)
                
    def forward(self, inputs):
        out = self.inp_layer(inputs)
        for block in self.main_block:
            out = block(out)
        out = self.out_layer(out)
        return out
    
    
class CoHiClustModel(nn.Module):
    def __init__(self, cfg, data_frame, embeds, n_clusters,
                 neighbor_mat_train=None, neighbor_mat_test=None):
        super().__init__()
             
        self.data_frame = data_frame
        self.clusters = data_frame["cluster"].to_numpy()
        self.embeds = embeds
        self.n_clusters = n_clusters
        
        self.f = VanillaMLP(cfg)
        
        self.g = nn.Sequential(
            nn.Linear(cfg.model.out_dim, 512, bias=False), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg.simclr.feature_dim_projection_head, bias=True)
        )
        self.tree_model = nn.Sequential(
            nn.Linear(
                cfg.model.out_dim,
                ((2 ** (cfg.tree.tree_level + 1)) - 1) - 2 ** cfg.tree.tree_level),
            nn.Sigmoid()
        )
        self.masks_for_level = {
            level: torch.ones(2 ** level).cuda() for level in range(1, cfg.tree.tree_level + 1)
        }
        
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        
        self.create_dataloaders(neighbor_mat_train, neighbor_mat_test)
        
    def forward(self, x):
        feature = self.f(x)
        out = self.g(feature)
        tree_output = self.tree_model(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), tree_output
    
    def create_dataloaders(
        self, neighbor_mat_train=None, neighbor_mat_test=None, k=5
    ):
        
        train_data = Banking1NNPair(
            self.data_frame, self.embeds, self.clusters, self.n_clusters, train=True, neighbor_mat=neighbor_mat_train, k=k
        )
        neighbor_mat = train_data.neighbor_mat

        memory_data = Banking1NNPair(
            self.data_frame, self.embeds, self.clusters, self.n_clusters, train=True, neighbor_mat=neighbor_mat, k=k
        )

        test_data = Banking1NNPair(
            self.data_frame, self.embeds, self.clusters, self.n_clusters, train=False, neighbor_mat=neighbor_mat_test, k=k
        )
        
        self.train_dataset = train_data
        self.memory_dataset = memory_data
        self.test_dataset = test_data

        self.train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
        )
        
        self.memory_loader = DataLoader(
            memory_data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        self.test_loader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
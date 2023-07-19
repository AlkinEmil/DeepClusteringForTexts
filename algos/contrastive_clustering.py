import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scipy.sparse import csr_matrix
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from typing import Tuple

from utils.cohiclust_utils import Banking1NNPair

class VanillaMLP(nn.Module):
    '''Vanilla MLP encoder for Contrastive Hierarchical CLustering model for text embeddings.'''
    def __init__(self, cfg: OmegaConf) -> None:
        '''Initialize ValillaMLP.
        
            :param cfg - config file with model parameters;
                         config.model should contain "inp_dim", "out_dim", "linear_dims_list" and "drop_prob"
        '''
        super().__init__()
              
        inp_dim = cfg.model.inp_dim
        out_dim = cfg.model.out_dim
        linear_dims_list = cfg.model.linear_dims_list
        drop_prob = cfg.model.drop_prob

        if not isinstance(inp_dim, int):
            raise TypeError("'inp_dim' must be integer")
        if not isinstance(out_dim, int):
            raise TypeError("'out_dim' must be integer")
        if not isinstance(linear_dims_list, ListConfig):
            raise TypeError("'linear_dims_list' must be list of ints")
        if not isinstance(drop_prob, float):
            raise TypeError("'drop_prob' must be float")
        
        if inp_dim <= 0:
            raise ValueError("'inp_dim' must be positive")
        if out_dim <= 0:
            raise ValueError("'out_dim' must be positive")
        if len(linear_dims_list) < 2:
            raise ValueError("Need at least 2 linear layers")
        if drop_prob < 0:
            raise ValueError("'drop_prob' must be non-negative")
        
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
                
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''Forward pass through the model.'''
        out = self.inp_layer(inputs)
        for block in self.main_block:
            out = block(out)
        out = self.out_layer(out)
        return out
    
    
class CoHiClustModel(nn.Module):
    '''Contrastive Hierarchical CLustering model. Model structure is from https://github.com/MichalZnalezniak/Contrastive-Hierarchical-Clustering'''
    def __init__(self,
                 cfg: OmegaConf,
                 data_frame: pd.DataFrame,
                 embeds: torch.tensor,
                 n_clusters: int,
                 neighbor_mat_train: csr_matrix = None,
                 neighbor_mat_test: csr_matrix = None
                ) -> None:
        '''Initialize Contrastive Hierarchical CLustering model.
        
            :param cfg - config file with model parameters (look in cfg folder for examples)
            :param data_frame - pandas.DataFrame containing texts (column "text") and cluster labels (columns "cluster")
            :param embeds - Tensor with text embeddings; len(embeds) should be equal to len(data_frame)
            :param n_clusters - number of clusters
            :param neighbor_mat_train - if not None, use pre-computed kNN matrix for the train set
            :param neighbor_mat_test - if not None, use pre-computed kNN matrix for the test set
        '''
        super().__init__()
        
        if not isinstance(data_frame, pd.DataFrame):
            raise TypeError("'data_frame' must be pandas.DataFrame")
        if not isinstance(embeds, torch.Tensor):
            raise TypeError("'embeds' must be torch.Tensor")
        if not isinstance(n_clusters, int):
            raise TypeError("'n_clusters' must be integer")
        if neighbor_mat_train is not None and not isinstance(neighbor_mat_train, csr_matrix):
            raise TypeError("'neighbor_mat_train' must be scipy.sparse.csr_matrix")
        if neighbor_mat_test is not None and not isinstance(neighbor_mat_test, csr_matrix):
            raise TypeError("'neighbor_mat_test' must be scipy.sparse.csr_matrix")
        if not isinstance(cfg.dataset.n_neighb, int):
            raise TypeError("'cfg.dataset.n_neighb' must be integer")
        
        if len(data_frame) != len(embeds):
            raise ValueError("number of rows in data_frame should be equal to number of text embeddings")
        if n_clusters <= 0:
            raise ValueError("'n_clusters' must be positive")
        if cfg.dataset.n_neighb<= 0:
            raise ValueError("'cfg.dataset.n_neighb' must be positive")
             
        self.data_frame = data_frame
        self.clusters = data_frame["cluster"].to_numpy()
        self.embeds = embeds
        self.n_clusters = n_clusters
        self.neighbor_mat_train = neighbor_mat_train
        self.neighbor_mat_test = neighbor_mat_test
        
        self.k = cfg.dataset.n_neighb
        
        # initalize backbone net
        self.f = VanillaMLP(cfg)
        
        # initialize 'output' net
        self.g = nn.Sequential(
            nn.Linear(cfg.model.out_dim, 512, bias=False), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg.simclr.feature_dim_projection_head, bias=True)
        )
        
        # initialize 'tree' net
        self.tree_model = nn.Sequential(
            nn.Linear(
                cfg.model.out_dim,
                ((2 ** (cfg.tree.tree_level + 1)) - 1) - 2 ** cfg.tree.tree_level),
            nn.Sigmoid()
        )
        
        # dictionary with level-leaf masks
        self.masks_for_level = {
            level: torch.ones(2 ** level).cuda() for level in range(1, cfg.tree.tree_level + 1)
        }
        
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        
        self.create_dataloaders()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Forward pass through the model.
        
            :param x - torch.Tensor of input embeddings
            :return a tuple of:
                * features (normalized output of self.f)
                * model output (normalized output of self.g(self.f))
                * tree_output (output of the self.tree_model)        
        '''
        feature = self.f(x)
        out = self.g(feature)
        tree_output = self.tree_model(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), tree_output
    
    def create_dataloaders(self):
        '''Create pair datasets and dataloaders for CoHiClust model.'''
        
        train_data = Banking1NNPair(
            self.data_frame, self.embeds, self.clusters, self.n_clusters, train=True,
            neighbor_mat=self.neighbor_mat_train, k=self.k
        )
        neighbor_mat = train_data.neighbor_mat

        memory_data = Banking1NNPair(
            self.data_frame, self.embeds, self.clusters, self.n_clusters, train=True,
            neighbor_mat=self.neighbor_mat_train, k=self.k
        )

        test_data = Banking1NNPair(
            self.data_frame, self.embeds, self.clusters, self.n_clusters, train=False,
            neighbor_mat=self.neighbor_mat_test, k=self.k
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
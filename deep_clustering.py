import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans

from utils.parametric_umap import NumpyToTensorDataset, FastTensorDataLoader, ContrastiveLoss

from annoy import AnnoyIndex
from scipy.sparse import lil_matrix


class DeepClustering(nn.Module):
    def __init__(self, n_classes, inp_dim, feat_dim, alpha, train_dataset, 
                 hid_dim=100,
                 loss_weights=None,
                 cluster_centers_init=None,
                 encoder=None,
                 decoder=None):
        '''
            n_classes: positive int - number of clusters
            inp_dim: positive int - dimension of the original space
            feat_dim: positive int - dimension of the feature space in which we do clustering
            alpha: float - parameter of the clustering loss
            hid_dim: positive int - dimension of the hidden space
            cluster_centers_init: torch.Tensor of shape (n_classes, hid_dim)
        '''
        super().__init__()
        assert isinstance(n_classes, int), "n_classes must be integer"
        assert isinstance(inp_dim, int), "inp_dim must be integer"
        assert isinstance(feat_dim, int), "feat_dim must be integer"
        assert isinstance(hid_dim, int), "hid_dim must be integer"
        
        assert n_classes > 0, "n_classes must be positive"
        assert inp_dim > 0, "inp_dim must be positive"
        assert feat_dim > 0, "feat_dim must be positive"
        assert hid_dim > 0, "hid_dim must be positive"
        
        self.embd_layer = nn.Embedding.from_pretrained(train_dataset, freeze=True)
        self.umap_loss = ContrastiveLoss(loss_mode="umap")
        
        self.K = n_classes
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.mode = "train_embeds"
        
        if loss_weights is None:
            loss_weights = [0.5, 0.5]
        else:
            assert isinstance(loss_weights, list), "loss_weights must be list"
            assert len(loss_weights) == 2
            assert isinstance(loss_weights[0], float)
            assert isinstance(loss_weights[1], float)
            #assert np.isclose(loss_weights[0] + loss_weights[1], 1)
            self.loss_weights = loss_weights
        
        if cluster_centers_init is None:
            self.centers = torch.nn.Parameter(torch.zeros(n_classes, feat_dim))
            torch.nn.init.xavier_uniform_(self.centers)
        else:
            assert isinstance(cluster_centers_init, torch.Tensor), "cluster_centers_init must be torch.Tensor"
            assert cluster_centers_init.shape == (n_classes, feat_dim)
            self.centers = nn.Parameter(cluster_centers_init)
        
        if encoder is None:
            self.enc = nn.Sequential(
                nn.Linear(inp_dim, hid_dim),
                #nn.BatchNorm1d(hid_dim),
                nn.ReLU(),
                #nn.Dropout(p=0.3),
                nn.Linear(hid_dim, hid_dim),
                #nn.BatchNorm1d(hid_dim),
                nn.ReLU(),
                #nn.Dropout(p=0.3),
                #nn.Linear(hid_dim, hid_dim),
                #nn.BatchNorm1d(hid_dim),
                #nn.ReLU(),
                #nn.Dropout(p=0.3),
                nn.Linear(hid_dim, feat_dim)
            )
        else:
            self.enc = encoder
        
        if decoder is None:
            self.dec = nn.Sequential(
                nn.Linear(feat_dim, hid_dim),
                #nn.BatchNorm1d(hid_dim),
                nn.ReLU(),
                #nn.Dropout(p=0.3),
                nn.Linear(hid_dim, hid_dim),
                #nn.BatchNorm1d(hid_dim),
                nn.ReLU(),
                #nn.Dropout(p=0.3),
                #nn.Linear(hid_dim, hid_dim),
                #nn.BatchNorm1d(hid_dim),
                #nn.ReLU(),
                #nn.Dropout(p=0.3),
                nn.Linear(hid_dim, inp_dim)
            )
        else:
            self.dec = decoder
        
    #def train_dataloader(self, neighbor_mat):
    #    train_dataloader = FastTensorDataLoader(
    #        self.neighbor_mat,
    #        shuffle=True,
    #        batch_size=self.bs,
    #        seed=0
    #     )
    #    return train_dataloader
        
        
    def train_clusters(self, x, loss_weights):
        assert isinstance(loss_weights, list), "loss_weights must be list"
        assert len(loss_weights) == 3
        assert isinstance(loss_weights[0], float)
        assert isinstance(loss_weights[1], float)
        assert isinstance(loss_weights[2], float)
        loss_weights = np.array(loss_weights)
        #assert np.isclose(loss_weights.sum(), 1)
        
        self.mode = "train_clusters"
        
        self.loss_weights = loss_weights
        z = self.enc(x)
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init="auto").fit(z.cpu().detach())
        cluster_centers_init = torch.tensor(kmeans.cluster_centers_, device=self.centers.device)
        self.centers = nn.Parameter(cluster_centers_init)
        return self

    def compute_q(self, z):
        assert z.shape[1] == self.feat_dim
        #x = self.enc(x)
        n = z.size(0)
        m = self.K
        a = z.unsqueeze(1).expand(n, m, self.feat_dim)
        b = self.centers.unsqueeze(0).expand(n, m, self.feat_dim)
        pairwise_distances = torch.pow(a - b, 2).sum(2) 
        #print("PD:", pairwise_distances)
        
        q_unnorm = torch.pow(pairwise_distances / self.alpha + 1, -(self.alpha+1)/2)
        q = q_unnorm / q_unnorm.sum(1, keepdim=True)
        return q
    
    def compute_clustering_loss(self, z):
        q = self.compute_q(z)
        f = q.sum(0, keepdim=True)
        p_unnorm = torch.pow(q, 3) / f
        p = p_unnorm / p_unnorm.sum(1, keepdim=True)
        kl_loss = nn.KLDivLoss(reduction='sum')#"batchmean")
        return kl_loss(torch.log(p), q)
    
    def compute_loss(self, item, neigh):
        neigh_input = torch.cat([item, neigh], dim=0)
        enc_neigh_input = self.enc(self.embd_layer(neigh_input))
        force_resample = 0
        geom_loss = self.umap_loss(enc_neigh_input, force_resample=force_resample)
        
        
        x = self.embd_layer(item)
        z = self.enc(x)
        if self.mode == "train_clusters":
            clustering_loss = self.compute_clustering_loss(z)
        x_recon = self.dec(z)
        recon_loss = F.mse_loss(x_recon, x)
        
        if self.mode == "train_embeds":
            total_loss = recon_loss * self.loss_weights[0] + geom_loss * self.loss_weights[1]
        else:
            total_loss = recon_loss * self.loss_weights[0] + \
                         geom_loss * self.loss_weights[1] + \
                         clustering_loss * self.loss_weights[2]
        loss = {
            "recon_loss": recon_loss,
            "geom_loss": geom_loss,
            "total_loss": total_loss
        }
        if self.mode == "train_clusters":
            loss["clustering_loss"] = clustering_loss
        return loss
    
    def transform(self, inputs, batch_size=None):
        device = inputs.device
        inputs = inputs.reshape(inputs.shape[0], -1)
        dataset_plain = NumpyToTensorDataset(inputs)
        dl_unshuf = torch.utils.data.DataLoader(
            dataset_plain,
            shuffle=False,
            batch_size=batch_size,
        )
        embd = np.vstack([self.enc(batch.to(device)).detach().cpu().numpy() for batch in dl_unshuf])

        return embd
    
    def transform_and_cluster(self, inputs, batch_size=None):
        device = inputs.device
        inputs = inputs.reshape(inputs.shape[0], -1)
        dataset_plain = NumpyToTensorDataset(inputs)
        dl_unshuf = torch.utils.data.DataLoader(
            dataset_plain,
            shuffle=False,
            batch_size=batch_size,
        )
        embds = np.vstack([self.enc(batch.to(device)).detach().cpu().numpy() for batch in dl_unshuf])
        centers = self.centers.cpu().detach().numpy()
        clusters = np.vstack([np.argmin(np.sum(np.power(embd - centers, 2), axis=1)) for embd in embds])
        return embds, clusters
    
    def create_dataloader(self, base_embeds, n_neighbours=40, annoy_trees=50, shuffle=True, batch_size=128, on_gpu=True):
        annoy = AnnoyIndex(self.inp_dim, "euclidean")
        [annoy.add_item(i, x) for i, x in enumerate(base_embeds)]
        annoy.build(annoy_trees)

        # construct the adjacency matrix for the graph
        adj = lil_matrix((base_embeds.shape[0], base_embeds.shape[0]))

        for i in range(base_embeds.shape[0]):
            neighs_, _ = annoy.get_nns_by_item(i, n_neighbours + 1, include_distances=True)
            neighs = neighs_[1:]
            adj[i, neighs] = 1
            adj[neighs, i] = 1  # symmetrize on the fly

        neighbor_mat = adj.tocsr()
        
        train_dataloader = FastTensorDataLoader(
           neighbor_mat,
           shuffle=shuffle,
           batch_size=batch_size,
           on_gpu=on_gpu,
        )
        return train_dataloader
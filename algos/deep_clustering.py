import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans

from annoy import AnnoyIndex
from scipy.sparse import lil_matrix

from typing import List, Dict, Tuple

from utils.parametric_umap import NumpyToTensorDataset, FastTensorDataLoader, ContrastiveLoss

class VanillaMLP(nn.Module):
    '''Vanilla MLP encoder/decoder for the DeepClustering model.'''
    def __init__(self, inp_dim: int, out_dim: int) -> None:
        '''Initialize VanillaMLP net.
        
            :param inp_dim - input dimensionality
            :param out_dim - output dimensionality
        '''
        super().__init__()
        
        if not isinstance(inp_dim, int):
            raise TypeError("'inp_dim' must be integer")
        if not isinstance(out_dim, int):
            raise TypeError("'out_dim' must be integer")
            
        if inp_dim <= 0:
            raise ValueError("'inp_dim' must be positive")
        if out_dim <= 0:
            raise ValueError("'out_dim' must be positive")
        
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, out_dim)
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''Forward pass through the model.'''
        return self.net(inputs)
    

class DeepClustering(nn.Module):
    '''Deep clustering model'''
    def __init__(self,
                 n_clusters: int, 
                 inp_dim: int, 
                 feat_dim: int, 
                 train_dataset: torch.Tensor,
                 alpha: float = 2,
                 loss_weights: List[float] = None,
                 cluster_centers_init: torch.Tensor = None,
                 deep_model_type: str = "DEC",
                 encoder: nn.Module = None,
                 decoder: nn.Module = None
                ) -> None:
        '''Initialize DeepClustering model.
        
            :param n_clusters: positive int - number of clusters
            :param inp_dim: positive int - dimension of the original space
            :param feat_dim: positive int - dimension of the feature space in which we do clustering
            :param train_dataset - (to do)
            :param alpha: float - parameter of the clustering loss
            :param loss_weights - list of weighting coefficients for losses (reconstruction, UMAP and clustering)
            :param cluster_centers_init - torch.Tensor of shape (n_clusters, hid_dim)
            :param deep_model_type: - "DEC", "DCN", "DEC+DCN", or "custom",
            :param encoder - if not None, use custom nn.Module as encoder
            :param decoder - if not None, use custom nn.Module as decoder
        '''
        super().__init__()
                
        if not isinstance(n_clusters, int):
            raise TypeError("'n_clusters' must be integer")
        if not isinstance(inp_dim, int):
            raise TypeError("'inp_dim' must be integer")
        if not isinstance(train_dataset, torch.Tensor):
            raise TypeError("'train_dataset' must be torch.Tensor")
        
        if (feat_dim is not None) and (not isinstance(feat_dim, int)):
            raise TypeError("'feat_dim' must be integer")
        
        if n_clusters <= 0:
            raise ValueError("'n_clusters' must be positive")
        if inp_dim <= 0:
            raise ValueError("'inp_dim' must be positive")
        if feat_dim <= 0:
            raise ValueError("'feat_dim' must be positive")
        
        self.embd_layer = nn.Embedding.from_pretrained(train_dataset, freeze=True)
        self.umap_loss = ContrastiveLoss(loss_mode="umap")
        
        if deep_model_type in ["DEC", "DCN", "DEC+DCN"]:
            self.deep_model_type = deep_model_type
        else:
            raise ValueError("deep_model_type `{}`".format(deep_model_type))
            
        self.n_clusters = n_clusters
        self.inp_dim = inp_dim
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
            self.loss_weights = loss_weights
        
        if cluster_centers_init is None:
            self.centers = torch.nn.Parameter(torch.zeros(n_clusters, feat_dim))
            torch.nn.init.xavier_uniform_(self.centers)
        else:
            if not isinstance(cluster_centers_init, torch.Tensor):
                raise TypeError("cluster centers must of type `torch.Tensor`")
            if cluster_centers_init.shape != (n_clusters, feat_dim):
                raise ValueError("cluster_centers_init must have shape ({}, {}), but not ({})"
                                 .format(n_clusters, feat_dim, tuple(cluster_centers_init.shape)))
            self.centers = nn.Parameter(cluster_centers_init)
        
        if encoder is None:
            self.enc = VanillaMLP(inp_dim, feat_dim)
        else:
            if not isinstance(encoder, nn.Module):
                raise TypeError("encoder must of type `torch.nn.Module`")
            try:
                if encoder.forward(torch.zeros(42, inp_dim)).shape != (42, feat_dim):
                    raise ValueError("encoder must be a map: R^{inp_dim} \\to R^{feat_dim}")
            except Exception:
                raise ValueError("encoder must be a map: R^{inp_dim} \\to R^{feat_dim}")
            self.enc = encoder
        
        if decoder is None:
            self.dec = VanillaMLP(feat_dim, inp_dim)
        else:
            if not isinstance(decoder, nn.Module):
                raise TypeError("decoder must of type `torch.nn.Module`")
            try:
                if decoder.forward(torch.zeros(42, feat_dim)).shape != (42, inp_dim):
                    raise ValueError("decoder must be a map: R^{feat_dim} \\to R^{inp_dim}")
            except Exception:
                raise ValueError("decoder must be a map: R^{feat_dim} \\to R^{inp_dim}")
            self.dec = decoder
        
    def train_clusters(self, x: torch.Tensor, loss_weights: List[float], random_state=None):
        '''Turn the model into "train_clusters" mode.
        
            :param x - input tensor of embeddings
            :param loss_weights - list of weight coefficients for losses (reconstruction, UMAP and clustering)
            :random_state - default value 'None'
        '''
        assert isinstance(loss_weights, dict), "loss_weights must be dict"
        assert isinstance(loss_weights["recon"], float)
        assert isinstance(loss_weights["geom"], float)
        
        if self.deep_model_type == "DEC":
            assert len(loss_weights) == 3
            assert isinstance(loss_weights["DEC"], float)
        elif self.deep_model_type == "DCN":
            assert len(loss_weights) == 4
            assert isinstance(loss_weights['inv_pw_dist'], float)
            assert isinstance(loss_weights['modified_DCN'], float)
        elif self.deep_model_type == "DEC+DCN":
            assert len(loss_weights) == 5
            assert isinstance(loss_weights["DEC"], float)
            assert isinstance(loss_weights['inv_pw_dist'], float)
            assert isinstance(loss_weights['modified_DCN'], float)
        else:
            raise ValueError("deep_model_type `{}`".format(self.deep_model_type))
        
        self.mode = "train_clusters"
        
        self.loss_weights = loss_weights
        z = self.enc(x)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state, n_init="auto").fit(z.cpu().detach())
        cluster_centers_init = torch.tensor(kmeans.cluster_centers_, device=self.centers.device)
        self.centers = nn.Parameter(cluster_centers_init)
        return self

    def compute_q(self, z: torch.Tensor) -> torch.Tensor:
        '''Compute surrogate q distribution.
        
            :param z - embeddings (output of encoder)
            :return q - surrogate cluster distribution
        '''
        assert z.shape[1] == self.feat_dim
        n = z.size(0)
        m = self.n_clusters
        a = z.unsqueeze(1).expand(n, m, self.feat_dim)
        b = self.centers.unsqueeze(0).expand(n, m, self.feat_dim)
        pairwise_distances = torch.pow(a - b, 2).sum(2) 
        
        q_unnorm = torch.pow(pairwise_distances / self.alpha + 1, -(self.alpha+1)/2)
        q = q_unnorm / q_unnorm.sum(1, keepdim=True)
        return q
    
    def compute_DEC_loss(self, z: torch.Tensor) -> torch.Tensor:
        '''Compute clustering (DEC) loss. See https://arxiv.org/abs/1511.06335
        
            :param z - embeddings (output of encoder)
            :return KL(p||q) - value of clustering loss function
        '''
        q = self.compute_q(z)
        f = q.sum(0, keepdim=True)
        p_unnorm = torch.pow(q, 3) / f
        p = p_unnorm / p_unnorm.sum(1, keepdim=True)
        kl_loss = nn.KLDivLoss(reduction='sum')
        return kl_loss(torch.log(p), q)
    
    def compute_inverse_pairwise_distance_loss(self) -> torch.Tensor:
        '''Compute regularization loss for DEC/DCN - inverse pairwise distance between current centroids and points.'''
        M = self.centers
        pdist = torch.nn.PairwiseDistance(p=2)
        loss = None
        for i in range(1, self.n_clusters):
            if loss is None:
                loss = torch.pow(pdist(M, torch.roll(M, i, 0)), -2).sum()
            else:
                loss += torch.pow(pdist(M, torch.roll(M, i, 0)), -2).sum()
        loss /= 2
        return loss
    
    def get_radius(self) -> torch.Tensor:
        '''Compute minimal cluster radius.'''
        M = self.centers
        pdist = torch.nn.PairwiseDistance(p=2)
        radius = None
        for i in range(1, self.n_clusters):
            if radius is None:
                radius = torch.pow(pdist(M, torch.roll(M, i, 0)), -2).min()
            else:
                radius = torch.min(torch.pow(pdist(M, torch.roll(M, i, 0)), -2).min(), radius)
        radius /= 3
        return radius
    
    
    def compute_modified_DCN_loss(self, item: torch.Tensor) -> torch.Tensor:
        '''Compute DCN loss.'''
        M = self.centers
        pdist = torch.nn.PairwiseDistance(p=2)
        loss = None
        for i in range(self.n_clusters):
            if loss is None:
                loss = torch.min(F.relu(pdist(M[i], item) - self.get_radius())**2)
            else:
                loss = loss + torch.min(F.relu(pdist(M[i], item) - self.get_radius())**2)
        return loss
    
    def compute_loss(self, item: torch.Tensor, neigh: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''Compute all losses.
        
            :param item - batch of text embedding
            :param neigh - batch of embedding of corresponding neighbor texts
            :return dictionary with losses: reconstruction MSE, geometric (UMAP) loss and DEC loss
        '''
        # compute geometric (UMAP) loss
        neigh_input = torch.cat([item, neigh], dim=0)
        enc_neigh_input = self.enc(self.embd_layer(neigh_input))
        force_resample = 0
        geom_loss = self.umap_loss(enc_neigh_input, force_resample=force_resample)
        
        # compute reconstruction loss
        x = self.embd_layer(item)
        z = self.enc(x)
        x_recon = self.dec(z)
        recon_loss = F.mse_loss(x_recon, x)
        if self.mode == "train_clusters":
            if self.deep_model_type == "DEC" or self.deep_model_type == "DEC+DCN":
                # compute DEC loss
                DEC_loss = self.compute_DEC_loss(z)
            if self.deep_model_type == "DCN" or self.deep_model_type == "DEC+DCN":
                # compute inverse distance between centers loss
                inv_pw_dist_loss = self.compute_inverse_pairwise_distance_loss()
                # compute modified DCN loss
                modified_DCN_loss = self.compute_modified_DCN_loss(z)
        
        loss = {
            "recon_loss": recon_loss,
            "geom_loss": geom_loss
        }
        
        if self.mode == "train_embeds":
            total_loss = recon_loss * self.loss_weights[0] + geom_loss * self.loss_weights[1]
        else:
            total_loss = recon_loss * self.loss_weights['recon'] + \
                         geom_loss * self.loss_weights['geom']
            if self.deep_model_type == "DEC" or self.deep_model_type == "DEC+DCN":
                total_loss = total_loss + DEC_loss * self.loss_weights['DEC']
                loss["DEC_loss"] = DEC_loss
            if self.deep_model_type == "DCN" or self.deep_model_type == "DEC+DCN":
                total_loss = total_loss + \
                             inv_pw_dist_loss * self.loss_weights['inv_pw_dist'] + \
                             modified_DCN_loss * self.loss_weights['modified_DCN']
                loss["inv_pw_dist_loss"] = inv_pw_dist_loss
                loss["modified_DCN_loss"] = modified_DCN_loss
        
        loss["total_loss"] = total_loss
        return loss
    
    def transform(self, inputs: np.array, batch_size: int = None) -> np.array:
        '''Apply encoder and transform text embeddings.
        
            :param inputs - numpy array with input text embeddings
            :param batch_size - batch size
            :return embd - transformed text embeddings
        '''
        device = inputs.device
        inputs = inputs.reshape(inputs.shape[0], -1)
        dataset_plain = NumpyToTensorDataset(inputs.detach().cpu().numpy())
        dl_unshuf = torch.utils.data.DataLoader(
            dataset_plain,
            shuffle=False,
            batch_size=batch_size,
        )
        embd = np.vstack([self.enc(batch.to(device)).detach().cpu().numpy() for batch in dl_unshuf])

        return embd
    
    def transform_and_cluster(self, inputs: np.array, batch_size: int = None) -> Tuple[np.array, np.array]:
        '''Trasform text embeddings and clusterize them.
        
            :param inputs - numpy array with input text embeddings
            :param batch_size - batch size
            :return a tuple of:
                * embd - transformed text embeddings
                * clusters - predicted cluster assignments
        '''
        embds = self.transform(inputs, batch_size)
        centers = self.centers.cpu().detach().numpy()
        clusters = np.vstack([np.argmin(np.sum(np.power(embd - centers, 2), axis=1)) for embd in embds])[:, 0]
        return embds, clusters
    
    def create_dataloader(
        self,
        base_embeds: np.array,
        n_neighbours: int = 40,
        annoy_trees: int = 50,
        shuffle: bool = True,
        batch_size: int = 128,
        on_gpu: bool = True
    ) -> DataLoader:
        '''Create dataloader of neighbor pairs for UMAP loss using annoy.
        
            :param base_embeds: numpy array with input text embeddings
            :param n_neighbours: number of nearest neighbour to find
            :param annoy_trees - parameter for approximate nearest neighbor search
            :param shuffle - if True, shuffle the dataset
            :param batch_size - batch size
            :param on_gpu - if True, move tensors to GPU
            :return train_dataloader of neighbor pairs
        '''
        annoy = AnnoyIndex(self.inp_dim, "euclidean")
        [annoy.add_item(i, x) for i, x in enumerate(base_embeds)]
        annoy.build(annoy_trees)

        # construct the adjacency matrix for the graph
        adj = lil_matrix((base_embeds.shape[0], base_embeds.shape[0]))

        for i in range(base_embeds.shape[0]):
            neighs_, _ = annoy.get_nns_by_item(i, n_neighbours + 1, include_distances=True)
            neighs = neighs_[1:]
            adj[i, neighs] = 1
            adj[neighs, i] = 1

        neighbor_mat = adj.tocsr()
        
        train_dataloader = FastTensorDataLoader(
           neighbor_mat,
           shuffle=shuffle,
           batch_size=batch_size,
           on_gpu=on_gpu,
        )
        
        return train_dataloader

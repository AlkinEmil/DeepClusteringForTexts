import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans


class DeepClustering(nn.Module):
    def __init__(self, n_classes, inp_dim, hid_dim, alpha, loss_weights=None, cluster_centers_init=None):
        '''
            n_classes: positive int - number of clusters
            inp_dim: positive int - dimension of the original space
            hid_dim: positive int - dimension of the hidden space in which we do clustering
            alpha: float - parameter of the clustering loss
            cluster_centers_init: torch.Tensor of shape (n_classes, hid_dim)
        '''
        super().__init__()
        assert isinstance(n_classes, int), "n_classes must be integer"
        assert isinstance(inp_dim, int), "inp_dim must be integer"
        assert isinstance(hid_dim, int), "n_classes must be integer"
        
        assert n_classes > 0, "n_classes must be positive"
        assert inp_dim > 0, "inp_dim must be positive"
        assert hid_dim > 0, "hid_dim must be positive"
        
        
        self.K = n_classes
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.alpha = alpha
        self.mode = "train_embeds"
        
        if loss_weights is None:
            loss_weights = [0.5, 0.5]
        else:
            assert isinstance(loss_weights, list), "loss_weights must be list"
            assert len(loss_weights) == 2
            assert isinstance(loss_weights[0], float)
            assert isinstance(loss_weights[1], float)
            assert np.isclose(loss_weights[0] + loss_weights[1], 1)
            self.loss_weights = loss_weights
        
        if cluster_centers_init is None:
            self.centers = torch.nn.Parameter(torch.zeros(n_classes, hid_dim))
            torch.nn.init.xavier_uniform_(self.centers)
        else:
            assert isinstance(cluster_centers_init, torch.Tensor), "cluster_centers_init must be torch.Tensor"
            assert cluster_centers_init.shape == (n_classes, hid_dim)
            self.centers = nn.Parameter(cluster_centers_init)
        
        self.enc = nn.Sequential(
            nn.Linear(inp_dim, inp_dim),
            nn.BatchNorm1d(inp_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(inp_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hid_dim, hid_dim)
        )
        
        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hid_dim, inp_dim),
            nn.BatchNorm1d(inp_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(inp_dim, inp_dim)
        )
        
        
    def train_clusters(self, x, loss_weights):
        assert isinstance(loss_weights, list), "loss_weights must be list"
        assert len(loss_weights) == 3
        assert isinstance(loss_weights[0], float)
        assert isinstance(loss_weights[1], float)
        assert isinstance(loss_weights[2], float)
        loss_weights = np.array(loss_weights)
        assert np.isclose(loss_weights.sum(), 1)
        
        self.mode = "train_clusters"
        
        self.loss_weights = loss_weights
        z = self.enc(x)
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init="auto").fit(z.cpu())
        cluster_centers_init = torch.tensor(kmeans.cluster_centers_, device=self.centers.device)
        self.centers = nn.Parameter(cluster_centers_init)
        return self

    def compute_q(self, z):
        assert z.shape[1] == self.hid_dim
        #x = self.enc(x)
        n = z.size(0)
        m = self.K
        a = z.unsqueeze(1).expand(n, m, self.hid_dim)
        b = self.centers.unsqueeze(0).expand(n, m, self.hid_dim)
        #print("a: ", a)
        #print("b: ", b)
        pairwise_distances = torch.pow(a - b, 2).sum(2) 
        #print("PD:", pairwise_distances)
        
        q_unnorm = torch.pow(pairwise_distances / self.alpha + 1, -(self.alpha+1)/2)
        q = q_unnorm / q_unnorm.sum(1, keepdim=True)
        return q
    
    def compute_clustering_loss(self, z):
        q = self.compute_q(z)
        f = q.sum(0, keepdim=True)
        p_unnorm = torch.pow(q, 2) / f
        p = p_unnorm / p_unnorm.sum(1, keepdim=True)
        kl_loss = nn.KLDivLoss(reduction='sum')#"batchmean")
        return kl_loss(torch.log(p), q)
    
    def compute_loss(self, x):
        z = self.enc(x)
        if self.mode == "train_clusters":
            clustering_loss = self.compute_clustering_loss(z)
        x_recon = self.dec(z)
        recon_loss = F.mse_loss(x_recon, x)
        geom_loss = torch.tensor(0)
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
    
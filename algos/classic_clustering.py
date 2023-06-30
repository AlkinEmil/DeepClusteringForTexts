import numpy as np

#import pandas as pd

from sklearn.cluster import KMeans
from umap.umap_ import UMAP


class ClassicClustering():
    def __init__(self, n_clusters, inp_dim, feat_dim, 
                 dim_reduction_type=None, clustering_type=None, 
                 random_state=None):
        '''
            n_clusters: positive int - number of clusters
            inp_dim: positive int - dimension of the original space
            feat_dim: positive int - dimension of the feature space in which we do clustering
        '''
        super().__init__()
        
        if not isinstance(n_clusters, int):
            raise TypeError("'n_clusters' must be integer")
        if not isinstance(inp_dim, int):
            raise TypeError("'inp_dim' must be integer")
        if not isinstance(feat_dim, int):
            raise TypeError("'feat_dim' must be integer")
        
        if n_clusters <= 0:
            raise ValueError("'n_clusters' must be positive")
        if inp_dim <= 0:
            raise ValueError("'inp_dim' must be positive")
        if feat_dim <= 0:
            raise ValueError("'feat_dim' must be positive")
        
        self.n_clusters = n_clusters
        self.inp_dim = inp_dim
        self.feat_dim = feat_dim
        
        if dim_reduction_type is None:
            self.dim_reduction = UMAP(random_state=random_state, n_components=feat_dim)
            
        if clustering_type is None:
            self.clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")

    
    def fit(self, inputs):
        self.dim_reduction.fit(inputs)
        embd = self.dim_reduction.transform(inputs)
        self.clustering.fit(embd)
        return self
    
    def transform_and_cluster(self, inputs):
        embds = self.dim_reduction.transform(inputs)
        clusters = self.clustering.predict(embds)
        return embds, clusters

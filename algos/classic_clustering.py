import numpy as np

#import pandas as pd

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
        self.dim_reduction_type = dim_reduction_type
        self.clustering_type = clustering_type
        
        if dim_reduction_type is None or dim_reduction_type == "umap":
            self.dim_reduction = UMAP(n_components=feat_dim, random_state=random_state)
        elif dim_reduction_type == "pca":
            self.dim_reduction = PCA(n_components=feat_dim, random_state=random_state)
        elif dim_reduction_type == "tsne" or dim_reduction_type == "tsne_barnes_hut":
            self.dim_reduction = TSNE(n_components=feat_dim, method='barnes_hut', random_state=random_state, n_jobs=-1)
        elif dim_reduction_type == "tsne_exact":
            self.dim_reduction = TSNE(n_components=feat_dim, method='exact', random_state=random_state, n_jobs=-1)
        else:
            raise ValueError("Unknown dim reduction method `{}`".format(dim_reduction_type))
            
        if clustering_type is None or clustering_type == "kmeans":
            self.clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        elif clustering_type == "spectral":
            self.clustering = SpectralClustering(n_clusters=n_clusters , random_state=random_state, n_jobs=-1)
        else:
            raise ValueError("Unknown clustering method `{}`".format(clustering_type))

    
    def fit(self, inputs):
        #self.dim_reduction.fit(inputs)
        embd = self.dim_reduction.fit_transform(inputs)
        if self.clustering_type != "spectral":
            self.clustering.fit(embd)
        return self
    
    def transform_and_cluster(self, inputs):
        if self.dim_reduction_type not in ["tsne", "tsne_exact", "tsne_barnes_hut"]:
            embds = self.dim_reduction.transform(inputs)
        else:
            embds = self.dim_reduction.fit_transform(inputs)
        if self.clustering_type != "spectral":
            clusters = self.clustering.predict(embds)
        else:
            clusters = self.clustering.fit_predict(embds)
        return embds, clusters

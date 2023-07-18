import numpy as np

from sklearn.cluster import KMeans, SpectralClustering, MeanShift
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from pacmap import PaCMAP

from typing import Tuple


class ClassicClustering():
    '''Wrapper for classic text clustering pipeline: dimensionality reduction + clustering.'''
    def __init__(self,
                 n_clusters: int,
                 inp_dim: int,
                 feat_dim: int = None,
                 dim_reduction_type: str = None,
                 clustering_type: str = None,
                 random_state: int = None,
                 min_samples: int = None,
                 min_cluster_size: int = None,
                 bandwidth: float = None
                ) -> None:
        '''Initialize classic clustering module.
        
            :param n_clusters: positive int - number of clusters
            :param inp_dim: positive int - dimension of the original space
            :param feat_dim: positive int - dimension of the feature space in which we do clustering
            :param dim_reduction_type: str or None - type of dimensionality reduction algorithm to be used; 
                                                     if None, use initial embeddings
            :param clustering_type: str - type of clustering algorithm to be used
            :param random_state: int or None - if not None, fix random state for reproducibility
            :param min_samples, min_cluster_size: positive ints - parameters of the HDBSCAN algorithm
            :param bandwidth: positive float - parameter of the MeanShift algorithm
        '''
        super().__init__()
        
        if not isinstance(n_clusters, int):
            raise TypeError("'n_clusters' must be integer")
        if not isinstance(inp_dim, int):
            raise TypeError("'inp_dim' must be integer")
        
        if n_clusters <= 0:
            raise ValueError("'n_clusters' must be positive")
        if inp_dim <= 0:
            raise ValueError("'inp_dim' must be positive")
        if dim_reduction_type is not None and feat_dim is None:
            raise ValueError("need to specify 'feat_dim' to use dimensionality reduction")
        if feat_dim is not None:
            if (not isinstance(feat_dim, int)) or (feat_dim <= 0):
                raise TypeError("'feat_dim' must be positive integer")
        
        self.n_clusters = n_clusters
        self.inp_dim = inp_dim
        self.feat_dim = feat_dim
        self.dim_reduction_type = dim_reduction_type
        self.clustering_type = clustering_type
        
        # initialize dimensionality reduction part
        if dim_reduction_type is None:
            self.dim_reduction = None     
        elif dim_reduction_type == "umap":
            self.dim_reduction = UMAP(n_components=feat_dim, random_state=random_state)
        elif dim_reduction_type == "pca":
            self.dim_reduction = PCA(n_components=feat_dim, random_state=random_state)
        elif dim_reduction_type == "tsne" or dim_reduction_type == "tsne_barnes_hut":
            self.dim_reduction = TSNE(n_components=feat_dim, method='barnes_hut', random_state=random_state, n_jobs=-1)
        elif dim_reduction_type == "tsne_exact":
            self.dim_reduction = TSNE(n_components=feat_dim, method='exact', random_state=random_state, n_jobs=-1)
        elif dim_reduction_type == "pacmap":
            self.dim_reduction = PaCMAP(n_components=feat_dim)
        else:
            raise ValueError("Unknown dim reduction method `{}`".format(dim_reduction_type))

        # initialize clustering part
        if clustering_type is None or clustering_type == "kmeans":
            self.clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        elif clustering_type == "spectral":
            self.clustering = SpectralClustering(n_clusters=n_clusters , random_state=random_state, n_jobs=-1)
        elif clustering_type == "hdbscan":
            assert min_samples is not None and min_cluster_size is not None, "Define HDBSCAN parameters"
            self.clustering = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
        elif clustering_type == "mean_shift":
            assert bandwidth is not None, "Define MeanShift parameter"
            self.clustering = MeanShift(bandwidth=bandwidth)
        elif clustering_type == "gmm":
            self.clustering = GaussianMixture(n_components=n_clusters, random_state=random_state)
        else:
            raise ValueError("Unknown clustering method `{}`".format(clustering_type))

    
    def fit(self, inputs: np.array) -> None:
        '''Fit dimensionality reduction method, if it is used.
        
            :param inputs: np.array with input text embeddings       
        '''
        if self.dim_reduction is not None:
            embd = self.dim_reduction.fit(inputs)
        return self
    
    def transform_and_cluster(self, inputs: np.array) -> Tuple[np.array, np.array]:
        '''Apply dimensionality reduction and clusterize embeddings.
        
            :param inputs: np.array with input text embeddings
            :return a tuple of:
                * embds - trasformed inputs (output of dimensionality reduction algorithm)
                * clusters - predicted clusters
        '''
        if self.dim_reduction_type is None:
            embds = inputs
        # currently, no 'transform' feature for pacmap
        elif self.dim_reduction_type == "pacmap":
            embds = self.dim_reduction.fit_transform(inputs)
        else:
            embds = self.dim_reduction.transform(inputs)
        
        # use fit-predict for versatility   
        clusters = self.clustering.fit_predict(embds)
        return embds, clusters

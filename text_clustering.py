import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from omegaconf import OmegaConf
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from typing import List, Any, Tuple, Dict

from algos.deep_clustering import DeepClustering
from algos.contrastive_clustering import CoHiClustModel
from algos.classic_clustering import ClassicClustering

from utils.training_and_visualisation import train
from utils.cohiclust_utils import train_cohiclust, test_cohiclust, RepeatPairDataset
from utils import topic_extraction

import umap.umap_ as umap
import matplotlib.pyplot as plt

class TextClustering(nn.Module):
    '''Wrapper for text clustering models.'''
    def __init__(self,
                 n_clusters: int,
                 inp_dim: int,
                 feat_dim: int,
                 train_dataset: torch.Tensor,
                 data_frame: pd.DataFrame,
                 cohiclust_cfg: OmegaConf = None,
                 loss_weights: List[float] = None,
                 cluster_centers_init: torch.Tensor = None,
                 encoder: nn.Module = None,
                 decoder: nn.Module = None,
                 kind: str = "deep clustering",
                 dim_reduction_type: str = None, 
                 clustering_type: str = None,
                 deep_model_type="DEC",
                 deep_params: dict = None,
                 random_state: int = None,
                 min_samples: int = None,
                 min_cluster_size: int = None,
                 bandwidth: float = None              
                ) -> None:
        '''Initialize TextClustering model.
        
            :param n_clusters: positive int - number of clusters
            :param inp_dim: positive int - dimension of the original space
            :param train_dataset - torch.Tensor of text embeddings (frozen)
            :param data_frame - pandas.DataFrame conraining texts and cluster markup
            :param cohiclust_cfg - OmegaConf with model parameters for the Contrastive Hierarchical Clustering model
            :param feat_dim - positive int - dimension of the feature space in which we do clustering
            :param loss_weights - list of weighting coefficients for DEC losses (reconstruction, UMAP and clustering)
            :param cluster_centers_init - torch.Tensor of shape (n_clusters, hid_dim)
            :param encoder - if not None, use custom nn.Module as encoder for DEC
            :param decoder - if not None, use custom nn.Module as decoder for DEC
            :param kind - model type (currently, "classic clustering", "deep clustering" and "cohiclust" are available)
            :param dim_reduction_type - type of dimensionality reduction algorithm to be used for classic clustering; 
                                        if None, use initial embeddings
            :param clustering_type - type of clustering algorithm to be used for classic clustering
            :param deep_model_type - deep model type ("DEC", "DCN", "DEC+DCN" and "custom" are available)
            :param deep_params - dict with model parameters for a deep clustering model
            :param random_state - if not None, fix random state for reproducibility
            :param min_samples, min_cluster_size - parameters of the HDBSCAN algorithm (in case of classic clustering)
            :param bandwidth - parameter of the MeanShift algorithm (in case of classic clustering)
        '''
        super().__init__()
                
        if not isinstance(n_clusters, int):
            raise TypeError("'n_clusters' must be integer")
        if not isinstance(inp_dim, int):
            raise TypeError("'inp_dim' must be integer")
        if feat_dim is not None and not isinstance(feat_dim, int):
            raise TypeError("'feat_dim' must be integer")
        if not isinstance(kind, str):
            raise TypeError("'kind' must be string")
        
        if n_clusters <= 0:
            raise ValueError("'n_clusters' must be positive")
        if inp_dim <= 0:
            raise ValueError("'inp_dim' must be positive")
        if feat_dim is not None and feat_dim <= 0:
            raise ValueError("'feat_dim' must be positive")
        
        self.n_clusters = n_clusters
        self.inp_dim = inp_dim
        self.feat_dim = feat_dim
        self.kind = kind
        self.data_frame = data_frame
        self.train_dataset = train_dataset
        self.times = {}
        self.deep_model_type = deep_model_type
        self.deep_params = deep_params
            
        if self.kind == "deep clustering":
            if encoder is not None and decoder is None:
                    raise ValueError("decoder must be not None")
            if decoder is not None and encoder is None:
                    raise ValueError("encoder must be not None")
                    
            self.model = DeepClustering(
                n_clusters=n_clusters,
                inp_dim=inp_dim,
                feat_dim=feat_dim,
                train_dataset=train_dataset,
                alpha=4, 
                loss_weights=[0.5, 0.5],
                deep_model_type=deep_model_type,
                encoder=encoder,
                decoder=decoder
            )
            
        elif self.kind == "classic clustering":
            self.model = ClassicClustering(
                n_clusters, 
                inp_dim, 
                feat_dim,
                dim_reduction_type=dim_reduction_type, 
                clustering_type=clustering_type, 
                random_state=random_state,
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                bandwidth=bandwidth
            )
            
        elif self.kind == "cohiclust":
            assert cohiclust_cfg is not None, "You must pass config for CoHiClust model."
            self.cohiclust_cfg = cohiclust_cfg
            
            self.model = CoHiClustModel(
                cohiclust_cfg,
                data_frame,
                train_dataset,
                n_clusters
            )
        
        else:
            raise ValueError(f"Wrong clustering type {self.kind}.")
        
    def fit(self, base_embeds: torch.Tensor, device: str = 'cuda') -> Any:
        '''Fit text Clustering model, measure time.
        
            :param base_embeds - text embeddings (frozen)
            :param device - device for training (in case of "deep clustering" or "cohiclust")
            :return training results or losses
        '''
        if self.kind == "deep clustering":
            N_ITERS_1 = 20
            LR_1 = 3e-3
            N_ITERS_2 = 8
            LR_2 = 1e-4
            LOSS_WEIGHTS = {
                "recon": 1.,
                "geom": 1.
            }
            if self.deep_model_type == "DEC" or self.deep_model_type == "DEC+DCN":
                LOSS_WEIGHTS["DEC"] = 1.
            if self.deep_model_type == "DCN" or self.deep_model_type == "DEC+DCN":
                LOSS_WEIGHTS["inv_pw_dist"] = 1.
                LOSS_WEIGHTS["modified_DCN"] = 1.
            
            if self.deep_params is not None:
                N_ITERS_1 = self.deep_params.get("N_ITERS_1", N_ITERS_1)
                LR_1 = self.deep_params.get("LR_1", LR_1)
                N_ITERS_2 = self.deep_params.get("N_ITERS_2", N_ITERS_2)
                LR_2 = self.deep_params.get("LR_2", LR_2)
                LOSS_WEIGHTS = self.deep_params.get("loss_weights", LOSS_WEIGHTS)
            
            self.model.to(device)

            optimizer = Adam(self.model.parameters(), lr=LR_1)
            print("Phase 1: train embeddings")
            
            start_time = time.time()
            losses1 = train(self.model, base_embeds, optimizer, N_ITERS_1, device)
            end_time = time.time()
            self.times["dim_red"] = end_time - start_time
            
            self.model.train_clusters(base_embeds.to(device), LOSS_WEIGHTS)
        
            
            # Change mode of the model to `train_clusters` and change weights of losses:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=LR_2)
            print("Phase 2: train clusters")
            
            start_time = time.time()
            losses2 = train(self.model, base_embeds, optimizer, N_ITERS_2, device)
            end_time = time.time()
            self.times["clust"] = end_time - start_time
            
            self.times["total"] = self.times["dim_red"] + self.times["clust"]
            
            return losses1, losses2
        
        elif self.kind == "classic clustering":
            start_time = time.time()
            self.model.fit(base_embeds)
            end_time = time.time()
            self.times["dim_red"] = end_time - start_time
            return None, None
        
        elif self.kind == "cohiclust":
            optimizer = Adam(self.model.parameters(), lr=1e-3)
            start_time = time.time()
            results =  train_cohiclust(
                self.model,
                optimizer,
                device=device
            )
            end_time = time.time()
            
            self.times["clust"] = end_time - start_time
            self.times["dim_red"] = 0
            self.times["total"] = self.times["dim_red"] + self.times["clust"]
            
            return results
        
    def get_centers(self) -> np.array:
        '''Get cluster centers.'''
        if self.kind == "deep clustering":
            return self.model.centers.cpu().detach().numpy()
        elif self.kind == "classic clustering":
            return self.model.clustering.cluster_centers_
        elif self.kind == "cohiclust":
            raise NotImplementedError("Clusters centers for cohiclust if are not implemented.")
   
    def visualize_2d(self, inputs, true_cluster_labels, random_state=None):
        if self.kind == "deep clustering":
            inputs = inputs.to(self.model.centers.device)
        dec_features, pred_clusters = self.transform_and_cluster(inputs)
        dim_reduction = umap.UMAP(random_state=random_state, n_components=2)
        umap_features = dim_reduction.fit_transform(dec_features)
        topics = self.get_topics(inputs)
        new_centers = dim_reduction.transform(self.get_centers())
        _, pred_clusters = self.transform_and_cluster(inputs)
        plt.figure(figsize=(9, 9))
        plt.scatter(*umap_features.T, c=true_cluster_labels, s=1.0)
        for i, (x, y) in enumerate(new_centers):
            if topics.get(i) is not None:
                plt.text(x, y, "\n".join(topics[i]), 
                         horizontalalignment='center', 
                         verticalalignment='center', 
                         fontsize=12)#, backgroundcolor='white')
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.show()
        
    def transform_and_cluster(self, inputs: torch.Tensor, batch_size: int = None) -> Tuple[torch.Tensor, np.array]:
        '''Transform initial text embeddings and clusterize them.
        
            :param inputs - torch.Tensor with initial embeddings
            :param batch_size - batch size for DeepClustering and CoHiClust models
            :return a tuple of:
                * transformed (encoded) text embeddings
                * predicted cluster assignments
        '''
        if self.kind == "deep clustering":
            inputs = inputs.to(self.model.centers.device)
            embeds, pred_clusters = self.model.transform_and_cluster(inputs, batch_size=batch_size)
        elif self.kind == "classic clustering":
            
            start_time = time.time()
            embeds, pred_clusters = self.model.transform_and_cluster(inputs)
            end_time = time.time()
            self.times["clust"] = end_time - start_time
            self.times["total"] = self.times["dim_red"] + self.times["clust"]
        
        elif self.kind == "cohiclust":
            if batch_size is None:
                batch_size = self.model.cfg.training.batch_size
            predict_dataset = RepeatPairDataset(inputs, self.data_frame["cluster"].to_list())
            predict_loader = DataLoader(predict_dataset, batch_size=batch_size)
            _, pred_clusters, labels = test_cohiclust(
                self.model, self.model.cfg.training.epochs, predict_loader, verbose=False
            )
            embeds = None
        
        # store predicted clusters for evaluation
        self.data_frame["pred_cluster"] = pred_clusters
            
        return embeds, pred_clusters
    
    def get_topics(self, inputs: torch.Tensor, language: str = "english") -> Dict[int, List[str]]:
        '''Clusterize texts and extract topics (key words) of predicted clusters.
        
            :param inputs -  torch.Tensor with initial embeddings
            :param language - language for topic extraction ("english" and "russial" are currently supported)
        '''
        assert language in ["english", "russian"], f"Not supported language {language}"
        
        if self.kind == "deep clustering":
            inputs = inputs.to(self.model.centers.device)
            if self.model.mode == "train_embeds":
                raise PermissionError("model must be in `train_clusters` mode")
        _, pred_clusters = self.transform_and_cluster(inputs)
        if language in ["english", "russian"]:
            topics_cluster_dict = topic_extraction.get_topics(self.data_frame, language=language)
        else:
            raise ValueError("Unknown language `{}`".format(lang))
        return topics_cluster_dict
    
    def evaluate(self, use_true_clusters: bool = False, language: str = "english", verbose: bool = True) -> Dict[str, float]:
        '''Evaluate TextClustering model.
        
            :param use_true_clusters - if True, compute markup-based metrics
            :param language - language for topic extraction ("english" and "russial" are currently supported)
            :param verbose - if True, print the results
            :return dictionary with metrics (silhouette score, adjusted Rand index, adjusted mutual information,
                                                                                                and average topic coherence)
        '''
        assert "pred_cluster" in self.data_frame.columns, "Predicted clusters are not in the dataframe. Call .predict() method"
        
        pred_clusters = self.data_frame["pred_cluster"].to_list()
        
        metrics = dict()
        
        if len(set(pred_clusters)) > 1:
            ss = silhouette_score(self.train_dataset, pred_clusters, metric='euclidean')
        else:
            ss = 0
        
        metrics["silhouette_score"] = ss
        
        if use_true_clusters:
            assert "cluster" in self.data_frame.columns, "True cluster labels are not in the dataframe"
            true_clusters = self.data_frame["cluster"].to_list()
            
            ars = adjusted_rand_score(true_clusters, pred_clusters)
            ams = adjusted_mutual_info_score(true_clusters, pred_clusters)
            
            metrics["adjusted_rand_score"] = ars
            metrics["adjusted_mutual_info_score"] = ams
            
        topics_cluster_dict = self.get_topics(self.train_dataset, language=language)
        topics_coh_dict = topic_extraction.compute_coherence_for_clusters(topics_cluster_dict, self.data_frame)
        
        avg_coh = np.mean(list(topics_coh_dict.values()))
        metrics["average_topic_coherence"] = avg_coh
        
        if verbose:
            for k, v in metrics.items():
                print(f"{k}: {np.round(v, 4)}")
        
        return metrics          
        

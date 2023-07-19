###########################################################################################################
# Code for Contrastive Hierarchical Clustering (model and losses) is taken from the oficial repository:   # 
# https://github.com/MichalZnalezniak/Contrastive-Hierarchical-Clustering                                 #
###########################################################################################################

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score

from typing import Dict, Tuple, List
from tqdm.notebook import tqdm
from annoy import AnnoyIndex
from scipy.sparse import lil_matrix, find, csr_matrix

#############################################################################
#                                  Dataset                                  #
#############################################################################

def compute_neighbour_matrix(k: int, features: torch.Tensor) -> csr_matrix:
    '''Compute approximate kNN matrix using annoy package.
    
        :param k - number of nearest neighbors to find
        :param features - torch.Tensor of shape (n_objects, n_features)
        :return binary scipy.sparse.csr_matrix with k nearest neighbors indicators
    '''
    obj_num, in_dim = features.shape
    annoy = AnnoyIndex(in_dim, "euclidean")
    [annoy.add_item(i, x) for i, x in enumerate(features)]
    annoy.build(50)

    # construct the adjacency matrix for the graph
    adj = lil_matrix((obj_num, obj_num))

    for i in range(features.shape[0]):
        neighs_, _ = annoy.get_nns_by_item(i, k + 1, include_distances=True)
        neighs = neighs_[1:]
        adj[i, neighs] = 1
        adj[neighs, i] = 1 

    neighbor_mat = adj.tocsr()
    return neighbor_mat

class Banking1NNPair(Dataset):
    '''Pair dataset for CoHiClust model. Use nearest neighbors to form positive pairs of objects.'''
    def __init__(
        self,
        base_data: pd.DataFrame,
        base_embeds: torch.Tensor,
        base_clusters: np.array,
        num_clusters: int,
        train: bool = True,
        neighbor_mat: csr_matrix = None,
        k: int = 5,
        seed: int = 42
    ) -> None:
        super().__init__()
        '''Initialize Banking1NNPair pair dataset.
        
            :param base_data - dataframe with original texts ("text" column) and markup ("cluster" column)
            :param base_embeds - initial texts embeddings
            :param base_clusters - true cluster assignments
            :param num_clusters - number of clusters
            :param train - if True, create train dataset
            :param neighbor_mat - if not None, use pre-computed neighbor matrix
            :param k - number of nearest neighbors to take into account
            :param seed - if not None, fix seed for reproducibility
        '''
        # train-test split
        embeds_train, embeds_test, clusters_train, clusters_test, data_train, data_test = train_test_split(
            base_embeds, base_clusters, base_data, test_size=0.3, random_state=seed
        )
        
        if train:
            self.embeds = embeds_train
            self.clusters = clusters_train
            self.data = data_train
        else:
            self.embeds = embeds_test
            self.clusters = clusters_test
            self.data = data_test
        
        self.k = k
        data_size = self.embeds.shape[0]
        
        # compute approximate kNN matrices. if necessary
        if neighbor_mat is None:
            print("Computing approximate KNN matrix")
            self.neighbor_mat = compute_neighbour_matrix(k=k, features=self.embeds.reshape(data_size, -1))
        else:
            assert (
                neighbor_mat.shape[0] == neighbor_mat.shape[1] and neighbor_mat.shape[0] == data_size
            ), "Shape of neighbor_mat must be (n_samples, n_samples)"
            
            self.neighbor_mat=neighbor_mat
        
        # fix seed
        random.seed(seed)
                    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        '''Get positive pair of text emebddings.'''
        emb, cluster = self.embeds[index], self.clusters[index]
        
        # find index of one of the k nearest neighbors
        i = random.choice(range(self.k))
        pair_index = find(self.neighbor_mat.getrow(index))[1][i]
        
        pos_1 = emb
        pos_2 = self.embeds[pair_index]

        return pos_1, pos_2, cluster
    
    def __len__(self) -> int:
        '''Return size of the dataset.'''
        return len(self.embeds)
    
class RepeatPairDataset(Dataset):
    '''Simulate pair dataset to make predictions for CoHiClust.'''
    def __init__(self, embeds: torch.Tensor, clusters: np.array) -> None:
        '''Initialize fake pair dataset.
        
            :param embeds - initial texts embeddings for prediction
            :param clusters - true clusters (not used in prediction)
        '''
        super().__init__()        
                
        self.embeds = embeds
        self.clusters = clusters
                    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        '''Return pair of copies of one object.'''
        emb, cluster = self.embeds[index], self.clusters[index]
        return emb, emb, cluster
    
    def __len__(self) -> int:
        '''Return size of the dataset.'''
        return len(self.embeds)
    
    
#############################################################################
#             Losses (code is copied from the oficial repo)                 #
#############################################################################

def probability_vec_with_level(feature: torch.Tensor, level: int) -> torch.Tensor:
    '''Compute tensor of probablities for a tree level.
    
        :param feature - output of the model
        :param level - number of the tree level
        :return probablities for a tree level
    '''
    prob_vec = torch.tensor([], requires_grad=True).cuda()
    for u in torch.arange(2 ** level - 1, 2 ** (level + 1) - 1, dtype=torch.long):
        probability_u = torch.ones_like(feature[:, 0], dtype=torch.float32).cuda()
        while(u > 0):
            if u / 2 > torch.floor(u / 2):
                # Go left
                u = torch.floor(u / 2) 
                u = u.long()
                probability_u *= feature[:, u]
            elif u / 2 == torch.floor(u / 2):
                # Go right
                u = torch.floor(u / 2) - 1
                u = u.long()
                probability_u *=  1 - feature[:, u]
        prob_vec = torch.cat((prob_vec, probability_u.unsqueeze(1)), dim=1)
    return prob_vec

def tree_loss(
    tree_output1: torch.Tensor,
    tree_output2: torch.Tensor,
    batch_size: int,
    mask_for_level: Dict[int, torch.Tensor],
    mean_of_probs_per_level_per_epoch: Dict[int, torch.Tensor],
    tree_level
):
    '''Compute loss function for the clustering head ('tree') of the model.
    
        :param tree_output1 - output of the tree model (predicted probabilities) for the 1st object in pair
        :param tree_output2 - output of the tree model (predicted probabilities) for the 2nd object in pair
        :param batch_size - batch size
        :param mask_for_level - dictionary with level-wise masks for a tree output
        :param mean_of_probs_per_level_per_epoch
        :param tree_level - depth of the tree
        :return value of the loss function
    '''
    ## TREE LOSS
    loss_value = torch.tensor([0], dtype=torch.float32, requires_grad=True).cuda()

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()
    
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels * ~mask
    out_tree = torch.cat([tree_output1, tree_output2], dim=0)

    for level in range(1, tree_level + 1):
        prob_features = probability_vec_with_level(out_tree, level)
        prob_features = prob_features * mask_for_level[level]
        if level == tree_level:
            mean_of_probs_per_level_per_epoch[tree_level] += torch.mean(prob_features, dim=0)
        # Calculate loss on positive classes
        loss_value -= torch.mean(
            torch.bmm(
                torch.sqrt(prob_features[torch.where(labels > 0)[0]].unsqueeze(1) +  1e-8),
                torch.sqrt(prob_features[torch.where(labels > 0)[1]].unsqueeze(2) + 1e-8)
            )
        )
        # Calculate loss on negative classes
        loss_value += torch.mean(
            torch.bmm(
                torch.sqrt(prob_features[torch.where(labels == 0)[0]].unsqueeze(1) + 1e-8),
                torch.sqrt(prob_features[torch.where(labels == 0)[1]].unsqueeze(2) + 1e-8)
            )
        )
    return loss_value

def regularization_loss(
    tree_output1: torch.Tensor,
    tree_output2: torch.Tensor, 
    masks_for_level: Dict[int, torch.Tensor],
    tree_level: int
) -> torch.Tensor:
    '''Compute regularization loss for CoHiClust model.
    
        :param tree_output1 - output of the tree model (predicted probabilities) for the 1st object in pair
        :param tree_output2 - output of the tree model (predicted probabilities) for the 2nd object in pair
        :param mask_for_level - dictionary with level-wise masks for a tree output
        :param tree_level - depth of the tree
        :return value of the regularization loss
    '''
    out_tree = torch.cat([tree_output1, tree_output2], dim=0)
    loss_reg = torch.tensor([0], dtype=torch.float32, requires_grad=True).cuda()
    
    for level in range(1, tree_level + 1):
        prob_features = probability_vec_with_level(out_tree, level)
        probability_leaves = torch.mean(prob_features, dim=0)
        probability_leaves_masked = masks_for_level[level] * probability_leaves
        for leftnode in range(0,int((2 ** level) / 2)):
            if not (masks_for_level[level][2 * leftnode] == 0 or masks_for_level[level][2 * leftnode + 1] == 0):
                loss_reg -= (
                    (1 / (2 ** level)) * 
                    (
                        0.5 * torch.log(probability_leaves_masked[2 * leftnode]) + 
                        0.5 * torch.log(probability_leaves_masked[2 * leftnode + 1])
                    )
                )
    return loss_reg
    

#############################################################################
#        Train & test (code is mostly copied from the oficial repo)         #
#############################################################################


def train_on_epoch(
    model: nn.Module,
    train_optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Train CoHiClust model for 1 epoch.
    
        :param model - CoHiCLust model to be trained
        :param train_optimizer - training optimizer from torch.optim
        :param epoch - number of the current training epoch
        :param device - if "cuda", use GPU for training
        :return a tuple of:
            * mean loss (total)
            * mean tree loss
            * mean contrastive loss
            * mean regularization loss
    '''
    batch_size = model.cfg.training.batch_size
    temperature = model.cfg.simclr.temperature
    tree_level =  model.cfg.tree.tree_level
    
    mean_of_probs_per_level_per_epoch = {
        level: torch.zeros(2 ** level).cuda() for level in range(1, tree_level + 1)
    }
 
    model.train()
    
    total_loss, total_num = 0.0, 0
    train_bar = tqdm(model.train_loader)
    total_tree_loss, total_reg_loss, total_simclr_loss = 0.0, 0.0, 0.0
    
    for pos_1, pos_2, _ in train_bar:
        pos_1 = pos_1.to(device)
        pos_2 = pos_2.to(device)
        
        feature_1, out_1, tree_output1 = model(pos_1)
        feature_2, out_2, tree_output2 = model(pos_2)
        out = torch.cat([out_1, out_2], dim=0)
        
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        
        loss_simclr = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        
        train_optimizer.zero_grad()
        if epoch > model.cfg.training.pretraining_epochs:
            tree_loss_value = tree_loss(
                tree_output1,
                tree_output2,
                batch_size,
                model.masks_for_level,
                mean_of_probs_per_level_per_epoch,
                tree_level
            )
            regularization_loss_value = regularization_loss(
                tree_output1,
                tree_output2,
                model.masks_for_level,
                tree_level
            )
            loss = loss_simclr + tree_loss_value + (2 ** (-model.cfg.tree.tree_level) * regularization_loss_value)
        else:
            loss = loss_simclr
            tree_loss_value = torch.zeros([1]) # Don't calculate the loss
            regularization_loss_value = torch.zeros([1]) # Don't calculate the loss 

        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_tree_loss += tree_loss_value.item() * batch_size
        total_reg_loss += regularization_loss_value.item() * batch_size
        total_simclr_loss += loss_simclr.item() * batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, model.cfg.training.epochs, total_loss / total_num)
        )

    if (
        epoch > model.cfg.training.start_pruning_epochs and 
        epoch <= model.cfg.training.start_pruning_epochs + model.cfg.training.leaves_to_prune
       ):
        x = mean_of_probs_per_level_per_epoch[tree_level] / len(model.train_loader)
        x = x.double()
        test = torch.where(x > 0.0, x, 1.0) 
        model.masks_for_level[tree_level][torch.argmin(test)] = 0
    return (
        total_loss / total_num, total_tree_loss / total_num,
        total_simclr_loss / total_num, total_reg_loss / total_num
    )


def test_cohiclust(
    model: nn.Module,
    epoch: int,
    test_loader: DataLoader = None,
    device: str = "cuda",
    verbose: bool = True
) -> Tuple[float, np.array, np.array]:
    '''Test CoHiCLust model.
    
        :param model - trained CoHiCLust model
        :param epoch - number of the current epoch (for logging)
        :param test_loader - DataLoader with test data (if None, use default dataloader for the model)
        :param device - if "cuda", use GPU for inference
        :param verbose - if True, print the result
        :return a tuple of:
            * normalized mutual information metric
            * predicted cluster assignments
            * true cluster labels
    '''
    if test_loader is None:
        test_loader = model.test_loader
        
    model.eval()
    
    tree_level =  model.cfg.tree.tree_level
    feature_bank = []
    labels, predictions = [], []
    
    with torch.no_grad():
        if hasattr(model.memory_loader.dataset, 'targets'):
            for data, _, target in tqdm(model.memory_loader, desc='Feature extracting'):
                feature, out, tree_output = model(data.to(device))
                feature_bank.append(feature)
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            feature_labels = torch.tensor(model.memory_loader.dataset.targets, device=feature_bank.device)
            
        test_bar = tqdm(test_loader)
        for data, _, target in test_bar:
            data = data.to(device)
            target = target.to(device)
            feature, out, tree_output = model(data)
            
            if hasattr(model.memory_loader.dataset, 'targets'):
                c = len(model.memory_loader.dataset.classes)
                sim_matrix = torch.mm(feature, feature_bank)
                sim_weight, sim_indices = sim_matrix.topk(k=model.cfg.simclr.k, dim=-1)
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / cfg.simclr.temperature).exp()

                one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                
                pred_scores = torch.sum(
                    one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1
                )
                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                
            prob_features = probability_vec_with_level(tree_output, tree_level)
            prob_features = model.masks_for_level[tree_level] * prob_features
            
            if hasattr(model.test_loader.dataset, 'subset_index_attr'):
                new_taget = []
                for elem in target:
                    new_taget.append(model.test_loader.dataset.subset_index_attr.index(elem))
                target = torch.Tensor(new_taget).to(dtype=torch.int64)
                
            for prediction, label in zip(torch.argmax(prob_features.detach(), dim=1), target.detach()):
                predictions.append(prediction.item())
                labels.append(label.item())
                
            actuall_nmi = normalized_mutual_info_score(labels, predictions)
            
            if verbose:
                test_bar.set_description(
                    'Test Epoch: [{}/{}] NMI:{:.2f}'.format(epoch, model.cfg.training.epochs, actuall_nmi)
                )

    return actuall_nmi, predictions, labels


def train_cohiclust(
    model,
    train_optimizer,
    device="cuda"
) -> Dict[str, List[float]]:
    '''Train CoHiCLust model and log the results.
    
        :param model - CoHiCLust model to be trained
        :param train_optimizer - training optimizer (from torch.optim)
        :param device - if "cuda", use GPU for training
        :return dictionary with training results (NMI and losses for epochs)
    '''
    results = {
        'train_loss': [],
        'tree_loss_train': [],
        'reg_loss_train' : [],
        'simclr_loss_train': [],
        'nmi': []
    }
    model.to(device)
    for epoch in range(1, model.cfg.training.epochs + 1):
        total_loss, tree_loss_train, reg_loss_train, simclr_loss_train = train_on_epoch(
            model, train_optimizer, epoch, device=device
        )
        results['train_loss'].append(total_loss)
        results['tree_loss_train'].append(tree_loss_train)
        results['reg_loss_train'].append(reg_loss_train)
        results['simclr_loss_train'].append(simclr_loss_train)
        
        if epoch > model.cfg.training.pretraining_epochs:
            nmi, _, _ = test_cohiclust(model, epoch, device=device)
            results['nmi'].append(nmi)
        else:
            results['nmi'].append(None)
            
    return results
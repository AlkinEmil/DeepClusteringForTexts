################################################################################################
# Code for parametric UMAP is from https://github.com/berenslab/contrastive-ne/tree/master     #
################################################################################################

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from annoy import AnnoyIndex
from scipy.sparse import lil_matrix


################################################################################################
#                                       Data utils                                             #
################################################################################################

class NumpyToTensorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, reshape=None):
        self.dataset = torch.tensor(dataset, dtype=torch.float32)
        if reshape is not None:
            self.reshape = lambda x: np.reshape(x, reshape)
        else:
            self.reshape = lambda x: x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        return self.reshape(item)

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """
    def __init__(self, neighbor_mat, batch_size=1024, shuffle=False, on_gpu=False, drop_last=False, seed=0):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :param on_gpu: If True, the dataset is loaded on GPU as a whole.
        :param drop_last: Drop the last batch if it is smaller than the others.
        :param seed: Random seed

        :returns: A FastTensorDataLoader.
        """

        neighbor_mat = neighbor_mat.tocoo()
        tensors = [torch.tensor(neighbor_mat.row),
                   torch.tensor(neighbor_mat.col)]
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)

        # manage device
        self.device = "cpu"
        if on_gpu:
            self.device="cuda"
            tensors = [tensor.to(self.device) for tensor in tensors]
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        torch.manual_seed(self.seed)

        # Calculate number of  batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0 and not self.drop_last:
            n_batches += 1
        self.n_batches = n_batches

        self.batch_size = torch.tensor(self.batch_size, dtype=int).to(self.device)

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i > self.dataset_len - self.batch_size:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
    
    
def compute_neighbour_matrix(k, features):
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
        adj[neighs, i] = 1  # symmetrize on the fly

    neighbor_mat = adj.tocsr()
    return neighbor_mat

    
################################################################################################
#                                          Loss                                                #
################################################################################################

def make_neighbor_indices(batch_size, negative_samples, device=None):
    """
    Selects neighbor indices
        :param batch_size: int Batch size
        :param negative_samples: int Number of negative samples
        :param device: torch.device Device of the model
        :return: torch.tensor Neighbor indices
    """
    b = batch_size

    if negative_samples < 2 * b - 2:
        # uniform probability for all points in the minibatch,
        # we sample points for repulsion randomly
        neg_inds = torch.randint(0, 2 * b - 1, (b, negative_samples), device=device)
        neg_inds += (torch.arange(1, b + 1, device=device) - 2 * b)[:, None]
    else:
        # full batch repulsion
        all_inds1 = torch.repeat_interleave(
            torch.arange(b, device=device)[None, :], b, dim=0
        )
        not_self = ~torch.eye(b, dtype=bool, device=device)
        neg_inds1 = all_inds1[not_self].reshape(b, b - 1)

        all_inds2 = torch.repeat_interleave(
            torch.arange(b, 2 * b, device=device)[None, :], b, dim=0
        )
        neg_inds2 = all_inds2[not_self].reshape(b, b - 1)
        neg_inds = torch.hstack((neg_inds1, neg_inds2))

    # now add transformed explicitly
    neigh_inds = torch.hstack(
        (torch.arange(b, 2 * b, device=device)[:, None], neg_inds)
    )

    return neigh_inds


class ContrastiveLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        negative_samples=5,
        temperature=0.07,
        loss_mode="umap",
        metric="euclidean",
        eps=1.0,
        clamp_high=1.0,
        clamp_low=1e-4,
        seed=0,
        loss_aggregation="mean",
    ):
        super(ContrastiveLoss, self).__init__()
        self.negative_samples = negative_samples
        self.temperature = temperature
        self.loss_mode = loss_mode
        self.metric = metric
        self.eps = eps
        self.clamp_high = clamp_high
        self.clamp_low = clamp_low
        self.seed = seed
        torch.manual_seed(self.seed)
        self.neigh_inds = None
        self.loss_aggregation = loss_aggregation

    def forward(self, features, log_Z=None, force_resample=False):
        """Compute loss for model. SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [2 * bsz, n_views, ...].
            log_Z: scalar, logarithm of the learnt normalization constant for nce.
            force_resample: Whether the negative samples should be forcefully resampled.
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0] // 2
        b = batch_size

        # We can at most sample this many samples from the batch.
        # `b` can be lower than `self.negative_samples` in the last batch.
        negative_samples = min(self.negative_samples, 2 * b - 1)

        if force_resample or self.neigh_inds is None:
            neigh_inds = make_neighbor_indices(
                batch_size, negative_samples, device=features.device
            )
            self.neigh_inds = neigh_inds
        # # untested logic to accomodate for last batch
        # elif self.neigh_inds.shape[0] != batch_size:
        #     neigh_inds = make_neighbor_indices(batch_size, negative_samples)
        #     # don't save this one
        else:
            neigh_inds = self.neigh_inds
        
        neighbors = features[neigh_inds]

        # `neigh_mask` indicates which samples feel attractive force and which ones repel each other
        neigh_mask = torch.ones_like(neigh_inds, dtype=torch.bool)
        neigh_mask[:, 0] = False

        origs = features[:b]

        # compute probits
        if self.metric == "euclidean":
            dists = ((origs[:, None] - neighbors) ** 2).sum(axis=2)
            # Cauchy affinities
            probits = torch.div(1, self.eps + dists)
        elif self.metric == "cosine":
            norm = torch.nn.functional.normalize
            o = norm(origs.unsqueeze(1), dim=2)
            n = norm(neighbors.transpose(1, 2), dim=1)
            logits = torch.bmm(o, n).squeeze() / self.temperature
            probits = torch.exp(logits)
        else:
            raise ValueError(f"Unknown metric “{self.metric}”")

        # compute loss    
        if self.loss_mode == "umap":
            # cross entropy parametric umap loss
            loss = -(~neigh_mask * torch.log(probits.clamp(self.clamp_low, self.clamp_high))) - (
                neigh_mask * torch.log((1 - probits).clamp(self.clamp_low, self.clamp_high))
            )
        else:
            raise ValueError(f"Unknown loss_mode “{self.loss_mode}”")

        # aggregate loss over batch
        if self.loss_aggregation == "sum":
            loss = loss.sum()
        else:
            loss = loss.mean()

        return loss
    

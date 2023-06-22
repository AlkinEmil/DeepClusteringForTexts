import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange
from IPython import display


def plot_losses(losses, title):
    plt.figure(figsize=(9, 6))
    plt.title(title)
    for i in losses:
        plt.plot(np.arange(len(losses[i])), losses[i], label=str(i))
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.show()
    
def plot_clusters_with_centers(model, base_embeds, clusters=None, title="Parametric UMAP for Banking77"):
    device = model.centers.device
    centers = model.centers.cpu().detach().numpy()
    if clusters is None:
        embeds, clusters = model.transform_and_cluster(base_embeds.to(device), batch_size=128)
    else:
        embeds = model.transform(base_embeds.to(device), batch_size=128)
    plt.figure(figsize=(9, 9))
    plt.scatter(*embeds.T, c=clusters, s=1.0)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', label='Cluster centers', s=200)
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(title)
    plt.legend()
    plt.show()
    
def train(model, data, optimizer, epochs, device="cpu", early_stoping=False, verbose=False):
    losses = dict()
    model.eval()
    dataloader = model.create_dataloader(data)
    _, init_clusters = model.transform_and_cluster(data.to(device))
    
    def update_losses(losses, loss):
        for i in loss:
            if losses.get(i) is None:
                losses[i] = [loss[i].item()]
            else:
                losses[i].append(loss[i].item())
    
    pbar = trange(epochs)
    for epoch in pbar:
        model.train()
        for batch in dataloader:
            #batch = batch.to(device)
            #x, neigh = batch
            x, neigh = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(x, neigh)
            update_losses(losses, loss)
            total_loss = loss["total_loss"]
            pbar.set_description(f"Total Loss: {total_loss.item():.2f}")
            total_loss.backward()
            optimizer.step()
            if verbose:
                plot_clusters_with_centers(model, data, clusters=init_clusters)
                display.clear_output(wait=True)
                #display.display(plt.gcf())
            if early_stoping and model.mode == "train_clusters" and np.mean(losses['clustering_loss'][-10:]) < np.mean(losses['geom_loss'][-10:]):
                return losses
    return losses
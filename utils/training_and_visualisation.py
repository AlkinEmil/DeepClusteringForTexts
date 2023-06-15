import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange


def plot_losses(losses, title):
    plt.figure(figsize=(9, 6))
    plt.title(title)
    for i in losses:
        plt.plot(np.arange(len(losses[i])), losses[i], label=str(i))
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.show()
    
    
def train(model, dataloader, optimizer, epochs, device="cpu"):
    losses = dict()
    
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
    return losses
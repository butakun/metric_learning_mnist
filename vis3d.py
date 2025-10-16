import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from feature_model import FeatureModel
from metrics import ArcFaceMetric

def main(ckp_filename):
    ckp = torch.load(ckp_filename)
    margin = ckp["margin"]
    scale = ckp["scale"]

    model = FeatureModel(out_size=3)
    model.load_state_dict(ckp["model"])
    model.eval()

    metric = ArcFaceMetric(10, 3, margin, scale)
    metric.load_state_dict(ckp["metric"])
    metric.eval()

    W = metric.W.detach().cpu().numpy()
    W = W / np.sqrt(np.square(W).sum(axis=1))[:, np.newaxis]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(W[:, 0], W[:, 1], W[:, 2], color="black")

    val_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    val_dataset = MNIST("./data", train=False, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    colors = ["blue","orange","green","purple","red","brown","pink","gray","olive","cyan"]

    for x, label in val_dataloader:
        with torch.no_grad():
            y = model(x)
        y = y.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        for i in range(10):
            mask = label == i
            ax.scatter(y[mask, 0], y[mask, 1], y[mask, 2], s=2, color=colors[i], alpha=0.2)

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    main(args.checkpoint)

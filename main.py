import time
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from feature_model import FeatureModel
from metrics import ArcFaceMetric

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"{device=}")

train_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
val_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def train(max_epoch, lr, batch_size, margin, scale):
    train_dataset = MNIST(root="./data", download=True, train=True, transform=train_transform)
    val_dataset = MNIST(root="./data", download=True, train=False, transform=val_transform)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = FeatureModel(out_size=3).to(device)
    metric = ArcFaceMetric(n_classes=10, latent_dim=3, margin=margin, scale=scale).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
            list(model.parameters()) + list(metric.parameters()),
            lr=lr
            )

    def save_checkpoint(filename):
        state_dict = {
                "model": model.state_dict(),
                "metric": metric.state_dict(),
                "optimizer": optimizer.state_dict(),
                "margin": margin,
                "scale": scale,
                }
        torch.save(state_dict, filename)

    for epoch in range(max_epoch):
        t0 = time.perf_counter()
        running_loss_train = 0.0
        n_train = 0
        for x, label in train_data_loader:
            x = x.to(device)
            label = label.to(device)

            y = model(x)
            logits = metric(y, label)
            loss = loss_function(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_batch = label.size(0)
            running_loss_train += loss.item() * n_batch
            n_train += n_batch

            if False:
                x_mean = x.view(x.size(0), -1).mean(1)
                print("*" * 80)
                print(f"{x_mean=}")
                print(f"{label=}")
                print(f"{y=}")
                print(f"{logits=}")
                print(f"{loss=}")
                print(f"{n_batch=}, {n_train=}")
        t1 = time.perf_counter()
        dt_epoch = t1 - t0

        running_loss_train /= n_train
        print(f"{epoch=}, {running_loss_train=}, {n_train=}, {dt_epoch=:5.2f}sec")

        if epoch % 100 == 0:
            save_checkpoint(f"checkpoint.{epoch}.pth")

    save_checkpoint(f"checkpoint.{max_epoch-1}.pth")


def main(args):
    train(args.max_epoch, lr=args.lr, batch_size=args.batch_size, margin=args.margin, scale=args.scale)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epoch", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1024*4)
    args = parser.parse_args()
    main(args)

import time
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from feature_model import create_model
from metrics import ArcFaceMetric, SoftmaxMetric

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"{device=}")

train_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
val_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

MODEL_TYPES = {
        "FeatureModelMLP": {
            "name": "FeatureModelMLP",
            "parameters": {
                "out_size": 2
                }
            },
        "FeatureModel2": {
            "name": "FeatureModel2", "parameters": {"out_size": 3},
            },
        "FeatureModel1": {
            "name": "FeatureModel1", "parameters": {"out_size": 3},
            },
        }


def train(model_name, metric_name, latent_dim, max_epoch, lr, batch_size, margin, scale):
    train_dataset = MNIST(root="./data", download=True, train=True, transform=train_transform)
    val_dataset = MNIST(root="./data", download=True, train=False, transform=val_transform)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model_config = MODEL_TYPES[model_name]
    model_config["parameters"]["out_size"] = latent_dim
    model = create_model(model_config).to(device)
    if metric_name == "ArcFace":
        metric = ArcFaceMetric(n_classes=10, latent_dim=latent_dim, margin=margin, scale=scale).to(device)
    elif metric_name == "Softmax":
        metric = SoftmaxMetric(n_classes=10, latent_dim=latent_dim, scale=scale).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
            list(model.parameters()) + list(metric.parameters()),
            lr=lr
            )

    def save_checkpoint(filename):
        state_dict = {
                "model_config": model_config,
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
        model.train()
        metric.train()
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

        t1 = time.perf_counter()
        dt_train_epoch = t1 - t0
        running_loss_train /= n_train

        with torch.no_grad():
            running_loss_val = 0.0
            n_val = 0
            model.eval()
            metric.eval()
            for x, label in val_data_loader:
                x = x.to(device)
                label = label.to(device)

                y = model(x)
                logits = metric(y, label)
                loss = loss_function(logits, label)

                n_batch = label.size(0)
                running_loss_val += loss.item() * n_batch
                n_val += n_batch

        t2 = time.perf_counter()
        dt_val_epoch = t2 - t1
        running_loss_val /= n_val

        print(f"{epoch=}, train_loss={running_loss_train}, val_loss={running_loss_val}, {n_train=}, train:{dt_train_epoch:5.2f}sec, val:{dt_val_epoch:5.2f}sec")

        if epoch % 100 == 0:
            save_checkpoint(f"checkpoint.{epoch}.pth")

    save_checkpoint(f"checkpoint.{max_epoch-1}.pth")


def main(args):
    train(
        args.model_name, args.metric_name, args.latent_dim,
        args.max_epoch,
        lr=args.lr, batch_size=args.batch_size, margin=args.margin, scale=args.scale
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="FeatureModel2")
    parser.add_argument("--metric-name", default="ArcFace", help="ArcFace|Softmax")
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--max-epoch", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1024*4)
    args = parser.parse_args()
    main(args)

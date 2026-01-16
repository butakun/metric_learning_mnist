from torch import nn
import torch.nn.functional as F


class FeatureModel(nn.Module):

    def __init__(self, img_size=28, hidden_size=64, out_size=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, out_size)
            )

    def forward(self, x):
        return F.normalize(self.model(x))


class FeatureModelMLP(nn.Module):

    def __init__(self, img_size=28, hidden_size=64, out_size=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, out_size)
            )

    def forward(self, x):
        return F.normalize(self.model(x))


class FeatureModel1(nn.Module):

    def __init__(self, out_size=2):
        super().__init__()
        # https://medium.com/@cr.tagadiya/arcface-loss-mnist-case-study-9ba89427d924
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(28 * 28 * 128, out_size)
            )

    def forward(self, x):
        return F.normalize(self.model(x))


class FeatureModel2(nn.Module):

    def __init__(self, out_size=3):
        super().__init__()

        # Model 2
        self.model = nn.Sequential(
            nn.Conv2d(1, 5, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(5, 5, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 5, out_size),
            )

    def forward(self, x):
        return F.normalize(self.model(x))


def create_model(config):
    if config["name"] == "FeatureModelMLP":
        return FeatureModelMLP(**config["parameters"])
    elif config["name"] == "FeatureModel1":
        return FeatureModel1(**config["parameters"])
    elif config["name"] == "FeatureModel2":
        return FeatureModel2(**config["parameters"])
    else:
        raise NotImplementedError(str(config))

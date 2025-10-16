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

        """
        # Model 2
        self.model = nn.Sequential(
            nn.Conv2d(1, 10, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(10, 20, 3, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Flatten(),
            nn.Linear(2880, out_size),
            )
        """

        """
        # Model 1
        # https://medium.com/@cr.tagadiya/arcface-loss-mnist-case-study-9ba89427d924
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(28 * 28 * 128, out_size)
            )
        """

    def forward(self, x):
        return F.normalize(self.model(x))

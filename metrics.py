import torch
from torch import nn
import torch.nn.functional as F


class ArcFaceMetric(nn.Module):

    def __init__(self, n_classes, latent_dim, margin, scale):
        super().__init__()

        self.n_classes = n_classes
        self.margin = margin
        self.scale = scale
        self.W = nn.Parameter(torch.empty((n_classes, latent_dim)), requires_grad=True)
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, label):
        # we assume a normalized x.
        cos_theta = F.linear(x, F.normalize(self.W, dim=1)).clamp(-0.999999, 0.999999)
        if False:
            W_mean = self.W.view(self.W.size(0), -1).mean(1)
            print(f"{W_mean=}")
            print(f"{cos_theta.min()=}, {cos_theta.max()=}")
        theta = torch.acos(cos_theta)
        theta = theta + F.one_hot(label, num_classes=self.n_classes) * self.margin
        cos_theta = torch.cos(theta)
        logits = self.scale * cos_theta
        return logits

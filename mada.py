import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class MADA(nn.Module):
    def __init__(self, n_classes, convnet=None, classifier=None):
        super().__init__()

        self._n_classes = n_classes

        self._convnet = convnet or ConvNet()
        self._classifier = classifier or Classifier(n_classes, 12544)
        self._grl = GRL(factor=-1)
        self._domain_classifiers = [
            Classifier(1, 12544)
            for _ in range(n_classes)
        ]

    def forward(self, x):
        features = self._convnet(x)
        features = features.view(features.shape[0], -1)

        logits = self._classifier(features)
        predictions = F.softmax(logits, dim=1)

        features = self._grl(features)
        domain_logits = []
        for class_idx in range(self._n_classes):
            weighted_features = predictions[:, class_idx].unsqueeze(1) * features
            domain_logits.append(
                self._domain_classifiers[class_idx](weighted_features)
            )

        return logits, domain_logits


class Classifier(nn.Module):
    def __init__(self, n_classes, input_dimension):
        super().__init__()

        self._n_classes = n_classes
        self._clf = nn.Linear(input_dimension, n_classes)

    def forward(self, x):
        return self._clf(x)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self._convnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self._convnet(x)


class GRL(torch.autograd.Function):
    def __init__(self, factor=-1):
        super().__init__()
        self._factor = factor

    def forward(self, x):
        return x

    def backward(self, grad):
        return self._factor * grad

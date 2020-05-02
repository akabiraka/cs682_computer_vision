import torchvision.models as models
import torch.nn as nn

import models.deeppose_config as config


class Deeppose(nn.Module):
    """
    using pretrained AlexNet
    """

    def __init__(self):
        super(Deeppose, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier = nn.Sequential(
            *list(alexnet.classifier.children())[:-1],
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(2048, config.n_joints * 2),
            nn.Sigmoid())

        self.alexnet = alexnet

    def forward(self, x):
        return self.alexnet(x)

import torch.nn as nn
import torch.nn.functional as F
import models.deeppose_config as config


class DeepposeLose(nn.Module):
    """docstring for DeepposeLose."""

    def __init__(self):
        super(DeepposeLose, self).__init__()

    def forward(self, out, target, meta=None):
        # meta = (target > -.5 + 1e-8).float().reshape(-1, config.n_joints, 2)
        out = out.reshape(-1, config.n_joints, 2)
        target = target.reshape(-1, config.n_joints, 2)
        loss = F.mse_loss(out, target)
        return loss

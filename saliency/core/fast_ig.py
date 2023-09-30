import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
class FastIG:
    def __init__(self, model):
        self.model = model

    def __call__(self, data, target):
        data.requires_grad_()
        # _,loss = self.model.get_loss(data, target)
        loss = F.cross_entropy(self.model(data), target)
        loss.backward()
        return (data * data.grad).detach().cpu().numpy()

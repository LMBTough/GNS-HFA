import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from torch.utils.data.sampler import Sampler
from torchvision import transforms
from PIL import Image

import json


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):

        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def pre_processing(obs, torch_device):

    obs = obs / 255

    obs = np.transpose(obs, (2, 0, 1))
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)

    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device)
    return obs_tensor


def fgsm_step(image, epsilon, data_grad_adv, data_grad_lab):

    grad_lab_norm = torch.norm(data_grad_lab, p=2)
    delta = epsilon * data_grad_adv.sign()

    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)
    delta = perturbed_rect - image
    delta = - data_grad_lab * delta
    return perturbed_rect, delta


def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()

    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):

        perturbed_image.requires_grad = True
        output = model(perturbed_image)

        pred = output.argmax(-1)

        for j in leave_index:
            if pred[j] == targeted[j]:
                leave_index.remove(j)
        if len(leave_index) == 0:
            break

        output = F.softmax(output, dim=1)

        loss = output[:, targeted].sum()

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        loss_lab = output[:, init_pred].sum()
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(
            image, epsilon, data_grad_adv, data_grad_lab)

        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    return c_delta, perturbed_image


def pgd_ssa_step(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()

    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):

        perturbed_image.requires_grad = True
        output = model(perturbed_image)

        pred = output.argmax(-1)

        for j in leave_index:
            if pred[j] == targeted[j]:
                leave_index.remove(j)
        if len(leave_index) == 0:
            break

        output = F.softmax(output, dim=1)

        loss = output[:, targeted].sum()

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        loss_lab = output[:, init_pred].sum()
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(
            image, epsilon, data_grad_adv, data_grad_lab)

        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    return c_delta, perturbed_image

class Dummy():
    pass


read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])


def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


def get_class_name(c):
    labels = json.load(open("imagenet_class_index.json"))

    return labels[str(c)][1]


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)

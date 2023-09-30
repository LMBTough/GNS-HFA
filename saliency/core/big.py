
import numpy as np
from captum.attr import IntegratedGradients
import torch
import torch.nn as nn
from .attack_methods import DI,gkern
import torch.nn.functional as F
from torch.autograd import Variable as V
from .dct import *
T_kernel = gkern(7, 3)
device = "cuda" if torch.cuda.is_available() else "cpu"

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

class FGSM:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon / 255
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=10, alpha=0.001):
        alpha = self.epsilon / num_steps
        dt = data.clone().detach().requires_grad_(True)
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            data_grad_sign = dt.grad.data.sign()
            adv_data = dt + alpha * data_grad_sign
            total_grad = adv_data - data
            total_grad = torch.clamp(
                total_grad, -self.epsilon, self.epsilon)
            dt.data = torch.clamp(
                data + total_grad, self.data_min, self.data_max)
        adv_pred = model(dt).argmax(-1)
        success = adv_pred != target
        return dt, success, adv_pred

T_kernel = gkern(7, 3)

class SSA:
    def __init__(self, epsilon, data_min, data_max):
        
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        
        self.image_width = 224
        self.momentum = 1
        self.num_iter = 10
        self.epsilon = epsilon / 255
        self.alpha = self.epsilon / self.num_iter
        self.rho = 0.5
        self.N = 20
        self.sigma = 16

    def __call__(self, model, data, target, num_steps=10, alpha=0.001):
        images_min = clip_by_tensor(data - self.epsilon, 0.0, 1.0)
        images_max = clip_by_tensor(data + self.epsilon, 0.0, 1.0)
        x = data.clone()
        grad = 0
        for i in range(self.num_iter):
            noise = 0
            for n in range(self.N):
                gauss = torch.randn(x.size()[0], 3, self.image_width, self.image_width) * (self.sigma / 255)
                gauss = gauss.cuda()
                x_dct = dct_2d(x + gauss).cuda()
                mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
                x_idct = idct_2d(x_dct * mask)
                x_idct = V(x_idct, requires_grad = True)

                output_v3 = model(DI(x_idct))
                loss = F.cross_entropy(output_v3, target)
                loss.backward()
                noise += x_idct.grad.data
            noise = noise / self.N

            noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

            noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            noise = self.momentum * grad + noise
            grad = noise

            x = x + self.alpha * torch.sign(noise)
            x = clip_by_tensor(x, images_min, images_max)
        x = x.detach()
        adv_pred = model(x).argmax(-1)
        success = adv_pred != target
        return x, success, adv_pred

    


def take_closer_bd(x, y, cls_bd, dis2cls_bd, boundary_points, boundary_labels):
    """Compare and return adversarial examples that are closer to the input

    Args:
        x (np.ndarray): Benign inputs
        y (np.ndarray): Labels of benign inputs
        cls_bd (None or np.ndarray): Points on the closest boundary
        dis2cls_bd ([type]): Distance to the closest boundary
        boundary_points ([type]): New points on the closest boundary
        boundary_labels ([type]): Labels of new points on the closest boundary

    Returns:
        (np.ndarray, np.ndarray): Points on the closest boundary and distances
    """
    if cls_bd is None:
        cls_bd = boundary_points
        dis2cls_bd = np.linalg.norm(np.reshape((boundary_points - x),
                                               (x.shape[0], -1)),
                                    axis=-1)
        return cls_bd, dis2cls_bd
    else:
        d = np.linalg.norm(np.reshape((boundary_points - x), (x.shape[0], -1)),
                           axis=-1)
        for i in range(cls_bd.shape[0]):
            if d[i] < dis2cls_bd[i] and y[i] != boundary_labels[i]:
                dis2cls_bd[i] = d[i]
                cls_bd[i] = boundary_points[i]
    return cls_bd, dis2cls_bd


def boundary_search(model, attacks, data, target, class_num=10,
                    num_steps=50, alpha=0.001):
    dis2cls_bd = np.zeros(data.shape[0]) + 1e16
    bd = None
    batch_boundary_points = None
    batch_success = None
    boundary_points = list()
    success_total = 0
    for attack in attacks:
        c_boundary_points, c_success, _ = attack(
            model, data, target, num_steps=num_steps, alpha=alpha)
        c_boundary_points = c_boundary_points
        batch_success = c_success
        success_total += torch.sum(batch_success.detach())
        if batch_boundary_points is None:
            batch_boundary_points = c_boundary_points.detach(
            ).cpu()
            batch_success = c_success.detach().cpu()
        else:
            for i in range(batch_boundary_points.shape[0]):
                if not batch_success[i] and c_success[i]:
                    batch_boundary_points[
                        i] = c_boundary_points[i]
                    batch_success[i] = c_success[i]
    boundary_points.append(batch_boundary_points)
    boundary_points = torch.cat(boundary_points, dim=0).to(device)
    y_pred = model(boundary_points).cpu().detach().numpy()
    x = data.cpu().detach().numpy()
    y = target.cpu().detach().numpy()
    y_onehot = np.eye(class_num)[y]
    bd, _ = take_closer_bd(x, y, bd,
                           dis2cls_bd, boundary_points.cpu(),
                           np.argmax(y_pred, -1))
    cls_bd = None
    dis2cls_bd = None
    cls_bd, dis2cls_bd = take_closer_bd(x, y_onehot, cls_bd,
                                        dis2cls_bd, bd, None)
    return cls_bd, dis2cls_bd, batch_success


class BIG:
    def __init__(self, model, attacks, class_num=10):
        self.model = model
        self.attacks = attacks
        self.class_num = class_num
        self.saliency = IntegratedGradients(model)

    def __call__(self, model, data, target, gradient_steps=50):
        cls_bd, _, success = boundary_search(
            model, self.attacks, data, target, self.class_num)
        attribution_map = self.saliency.attribute(
            data, target=target, baselines=cls_bd.to(device), n_steps=gradient_steps, method="riemann_trapezoid")
        return attribution_map.cpu().detach().numpy(), success

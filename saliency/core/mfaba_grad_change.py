import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .attack_methods import DI,gkern
from torch.autograd import Variable as V
from .dct import *
from scipy import stats as st
features = None


def hook_feature(module, input, output):
    global features
    features = output


def image_transform(x):
    return x


def get_NAA_loss(adv_feature, base_feature, weights):
    gamma = 1.0
    attribution = (adv_feature - base_feature) * weights
    blank = torch.zeros_like(attribution)
    positive = torch.where(attribution >= 0, attribution, blank)
    negative = torch.where(attribution < 0, attribution, blank)
    # Transformation: Linear transformation performs the best
    balance_attribution = positive + gamma * negative
    loss = torch.sum(balance_attribution) / \
        (base_feature.shape[0]*base_feature.shape[1])
    return loss


def normalize(grad, opt=2):
    if opt == 0:
        nor_grad = grad
    elif opt == 1:
        abs_sum = torch.sum(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        nor_grad = grad/abs_sum
    elif opt == 2:
        square = torch.sum(torch.square(grad), dim=(1, 2, 3), keepdim=True)
        nor_grad = grad/torch.sqrt(square)
    return nor_grad


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / \
        (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, kern_size):
    x = torch.nn.functional.pad(
        x, (kern_size, kern_size, kern_size, kern_size), "constant", 0)
    x = torch.nn.functional.conv2d(
        x, stack_kern, stride=1, padding=0, groups=3)
    return x


"""Input diversity: https://arxiv.org/abs/1803.06978"""


def input_diversity(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize,
                        size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(
        x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(),
                            size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(),
                             size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(
    ), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret


P_kern, kern_size = project_kern(3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_kernel = gkern(7, 3)


class FGSMGrad:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        alpha = self.epsilon / num_steps
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            if use_softmax:
                tgt_out = torch.diag(
                    F.softmax(output, dim=-1)[:, target]).unsqueeze(-1)
            else:
                tgt_out = torch.diag(output[:, target]).unsqueeze(-1)
            tgt_out.sum().backward()
            grad = dt.grad.detach()
            temp_output = model(dt)
            model.zero_grad()
            if use_softmax:
                temp_output = F.softmax(temp_output, dim=-1)
            loss = F.cross_entropy(temp_output, target)
            loss.backward()
            temp_grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(temp_grad[i:i+1].clone())
            if use_sign:
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        adv_pred = model(dt)
        model.zero_grad()
        if use_softmax:
            tgt_out = torch.diag(F.softmax(adv_pred, dim=-1)
                                 [:, target_clone]).unsqueeze(-1)
        else:
            tgt_out = torch.diag(adv_pred[:, target_clone]).unsqueeze(-1)
        tgt_out.sum().backward()
        grad = dt.grad.detach()
        temp_output = model(dt)
        model.zero_grad()
        if use_softmax:
            temp_output = F.softmax(temp_output, dim=-1)
        loss = F.cross_entropy(temp_output, target_clone)
        loss.backward()
        temp_grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(temp_grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads



class FGSMGradSSA:
    def __init__(self, epsilon, data_min, data_max,N=20):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.N = N

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        image_width = 224
        momentum = 1.0
        alpha = self.epsilon / num_steps
        grad = 0
        rho = 0.5
        N = self.N
        sigma = 16
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        for _ in tqdm(range(num_steps)):
            model.zero_grad()
            noise = 0
            for n in tqdm(range(N)):
                gauss = torch.randn(dt.size()[0], 3, image_width, image_width) * (sigma / 255)
                gauss = gauss.cuda()
                dt_dct = dct_2d(dt + gauss).cuda()
                mask = (torch.rand_like(dt) * 2 * rho + 1 - rho).cuda()
                dt_idct = idct_2d(dt_dct * mask)
                dt_idct = V(dt_idct, requires_grad = True)

                # DI-FGSM https://arxiv.org/abs/1803.06978
                output_v3 = model(DI(dt_idct))
                if use_softmax:
                    output_v3 = F.softmax(output_v3, dim=-1)
                loss = F.cross_entropy(output_v3, target)
                loss.backward()
                noise += dt_idct.grad.data
            noise = noise / N
            # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
            noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

            # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
            noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            noise = momentum * grad + noise
            grad = noise
            grad = grad.detach()
            temp_output = model(dt)
            model.zero_grad()
            if use_softmax:
                temp_output = F.softmax(temp_output, dim=-1)
            loss = F.cross_entropy(temp_output, target)
            loss.backward()
            temp_grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(temp_grad[i:i+1].clone())
            if use_sign:
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
                    grad = grad[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        adv_pred = model(dt)
        model.zero_grad()
        grad = 0
        noise = 0
        for n in range(N):
            gauss = torch.randn(dt.size()[0], 3, image_width, image_width) * (sigma / 255)
            gauss = gauss.cuda()
            dt_dct = dct_2d(dt + gauss).cuda()
            mask = (torch.rand_like(dt) * 2 * rho + 1 - rho).cuda()
            dt_idct = idct_2d(dt_dct * mask)
            dt_idct = V(dt_idct, requires_grad = True)

            # DI-FGSM https://arxiv.org/abs/1803.06978
            output_v3 = model(DI(dt_idct))
            if use_softmax:
                output_v3 = F.softmax(output_v3, dim=-1)
            loss = F.cross_entropy(output_v3, target_clone)
            loss.backward()
            noise += dt_idct.grad.data
        noise = noise / N
        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        grad = grad.detach()
        temp_output = model(dt)
        model.zero_grad()
        if use_softmax:
            temp_output = F.softmax(temp_output, dim=-1)
        loss = F.cross_entropy(temp_output, target_clone)
        loss.backward()
        temp_grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(temp_grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads


class PGDGrad:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        alpha = self.epsilon / num_steps
        dt = data.clone().detach()
        dt = dt + torch.empty_like(dt).uniform_(-self.epsilon, self.epsilon)
        dt = torch.clamp(dt, min=self.data_min, max=self.data_max).detach()
        dt = dt.requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            if use_softmax:
                tgt_out = torch.diag(
                    F.softmax(output, dim=-1)[:, target]).unsqueeze(-1)
            else:
                tgt_out = torch.diag(output[:, target]).unsqueeze(-1)
            tgt_out.sum().backward()
            grad = dt.grad.detach()
            temp_output = model(dt)
            model.zero_grad()
            if use_softmax:
                temp_output = F.softmax(temp_output, dim=-1)
            loss = F.cross_entropy(temp_output, target)
            loss.backward()
            temp_grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(temp_grad[i:i+1].clone())
            if use_sign:
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        adv_pred = model(dt)
        model.zero_grad()
        if use_softmax:
            tgt_out = torch.diag(F.softmax(adv_pred, dim=-1)
                                 [:, target_clone]).unsqueeze(-1)
        else:
            tgt_out = torch.diag(adv_pred[:, target_clone]).unsqueeze(-1)
        tgt_out.sum().backward()
        grad = dt.grad.detach()
        temp_output = model(dt)
        model.zero_grad()
        if use_softmax:
            temp_output = F.softmax(temp_output, dim=-1)
        loss = F.cross_entropy(temp_output, target_clone)
        loss.backward()
        temp_grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(temp_grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads



class DIFGSMGrad:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.decay=0.0
        self.resize_rate=0.9
        self.diversity_prob=0.5
        
    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        alpha = self.epsilon / num_steps
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        momentum = 0
        leave_index = np.arange(data.shape[0])
        for _ in range(num_steps):
            output = model(self.input_diversity(dt))
            model.zero_grad()
            if use_softmax:
                tgt_out = torch.diag(
                    F.softmax(output, dim=-1)[:, target]).unsqueeze(-1)
            else:
                tgt_out = torch.diag(output[:, target]).unsqueeze(-1)
            tgt_out.sum().backward()
            grad = dt.grad.detach()
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad
            temp_output = model(dt)
            model.zero_grad()
            if use_softmax:
                temp_output = F.softmax(temp_output, dim=-1)
            loss = F.cross_entropy(temp_output, target)
            loss.backward()
            temp_grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(temp_grad[i:i+1].clone())
            if use_sign:
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
                    momentum = momentum[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        adv_pred = model(self.input_diversity(dt))
        model.zero_grad()
        if use_softmax:
            tgt_out = torch.diag(F.softmax(adv_pred, dim=-1)
                                 [:, target_clone]).unsqueeze(-1)
        else:
            tgt_out = torch.diag(adv_pred[:, target_clone]).unsqueeze(-1)
        tgt_out.sum().backward()
        grad = dt.grad.detach()
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        temp_output = model(dt)
        model.zero_grad()
        if use_softmax:
            temp_output = F.softmax(temp_output, dim=-1)
        loss = F.cross_entropy(temp_output, target_clone)
        loss.backward()
        temp_grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(temp_grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads



class TIFGSMGrad:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.decay=0.0
        self.kernel_name='gaussian'
        self.len_kernel=15
        self.nsig=3
        self.resize_rate=0.9
        self.diversity_prob=0.5
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.device = "cuda"
        
    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        alpha = self.epsilon / num_steps
        dt = data.clone().detach().requires_grad_(True)
        momentum = 0
        stacked_kernel = self.stacked_kernel.to(self.device)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        for _ in range(num_steps):
            output = model(self.input_diversity(dt))
            model.zero_grad()
            if use_softmax:
                tgt_out = torch.diag(
                    F.softmax(output, dim=-1)[:, target]).unsqueeze(-1)
            else:
                tgt_out = torch.diag(output[:, target]).unsqueeze(-1)
            tgt_out.sum().backward()
            grad = dt.grad.detach()
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad
            temp_output = model(dt)
            model.zero_grad()
            if use_softmax:
                temp_output = F.softmax(temp_output, dim=-1)
            loss = F.cross_entropy(temp_output, target)
            loss.backward()
            temp_grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(temp_grad[i:i+1].clone())
            if use_sign:
                
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
                    momentum = momentum[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        adv_pred = model(self.input_diversity(dt))
        model.zero_grad()
        if use_softmax:
            tgt_out = torch.diag(F.softmax(adv_pred, dim=-1)
                                 [:, target_clone]).unsqueeze(-1)
        else:
            tgt_out = torch.diag(adv_pred[:, target_clone]).unsqueeze(-1)
        tgt_out.sum().backward()
        grad = dt.grad.detach()
        grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        temp_output = model(dt)
        model.zero_grad()
        if use_softmax:
            temp_output = F.softmax(temp_output, dim=-1)
        loss = F.cross_entropy(temp_output, target_clone)
        loss.backward()
        temp_grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(temp_grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads

class MIFGSMGrad:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.decay = 1

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        alpha = self.epsilon / num_steps
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        momentum = 0
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            if use_softmax:
                tgt_out = torch.diag(
                    F.softmax(output, dim=-1)[:, target]).unsqueeze(-1)
            else:
                tgt_out = torch.diag(output[:, target]).unsqueeze(-1)
            tgt_out.sum().backward()
            grad = dt.grad.detach()
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad
            temp_output = model(dt)
            model.zero_grad()
            if use_softmax:
                temp_output = F.softmax(temp_output, dim=-1)
            loss = F.cross_entropy(temp_output, target)
            loss.backward()
            temp_grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(temp_grad[i:i+1].clone())
            if use_sign:
                
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
                    momentum = momentum[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        adv_pred = model(dt)
        model.zero_grad()
        if use_softmax:
            tgt_out = torch.diag(F.softmax(adv_pred, dim=-1)
                                 [:, target_clone]).unsqueeze(-1)
        else:
            tgt_out = torch.diag(adv_pred[:, target_clone]).unsqueeze(-1)
        tgt_out.sum().backward()
        grad = dt.grad.detach()
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        temp_output = model(dt)
        model.zero_grad()
        if use_softmax:
            temp_output = F.softmax(temp_output, dim=-1)
        loss = F.cross_entropy(temp_output, target)
        loss.backward()
        temp_grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(temp_grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads


class SINIFGSMGrad:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.decay = 1
        self.m = 5
        self.device = "cuda"

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        alpha = self.epsilon / num_steps
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        momentum = 0
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        for _ in range(num_steps):
            nes_dt = dt + self.decay*alpha*momentum
            adv_grad = torch.zeros_like(dt).to(self.device)
            for i in torch.arange(self.m):
                nes_images = nes_dt / torch.pow(2, i)
                outputs = model(nes_images)
                if use_softmax:
                    tgt_out = torch.diag(
                        F.softmax(outputs, dim=-1)[:, target]).unsqueeze(-1)
                else:
                    tgt_out = torch.diag(outputs[:, target]).unsqueeze(-1)
                cost = tgt_out.sum()
                adv_grad += torch.autograd.grad(cost, dt,
                                                retain_graph=False, create_graph=False)[0]
            adv_grad = adv_grad / self.m

            # Update adversarial images
            grad = self.decay*momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(1,2,3), keepdim=True)
            momentum = grad
            # output = model(dt)
            # model.zero_grad()
            # if use_softmax:
            #     tgt_out = torch.diag(
            #         F.softmax(output, dim=-1)[:, target]).unsqueeze(-1)
            # else:
            #     tgt_out = torch.diag(output[:, target]).unsqueeze(-1)
            # tgt_out.sum().backward()
            # grad = dt.grad.detach()
            temp_output = model(dt)
            model.zero_grad()
            if use_softmax:
                temp_output = F.softmax(temp_output, dim=-1)
            loss = F.cross_entropy(temp_output, target)
            loss.backward()
            temp_grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(temp_grad[i:i+1].clone())
            if use_sign:
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        nes_dt = dt + self.decay*alpha*momentum
        adv_grad = torch.zeros_like(dt).to(self.device)
        for i in torch.arange(self.m):
            nes_images = nes_dt / torch.pow(2, i)
            outputs = model(nes_images)
            if use_softmax:
                tgt_out = torch.diag(
                    F.softmax(outputs, dim=-1)[:, target]).unsqueeze(-1)
            else:
                tgt_out = torch.diag(outputs[:, target]).unsqueeze(-1)
            cost = tgt_out.sum()
            adv_grad += torch.autograd.grad(cost, dt,
                                            retain_graph=False, create_graph=False)[0]
        adv_grad = adv_grad / self.m

        # Update adversarial images
        grad = self.decay*momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(1,2,3), keepdim=True)
        adv_pred = model(dt)
        temp_output = model(dt)
        model.zero_grad()
        if use_softmax:
            temp_output = F.softmax(temp_output, dim=-1)
        loss = F.cross_entropy(temp_output, target)
        loss.backward()
        temp_grad = dt.grad.detach()
        # model.zero_grad()
        # if use_softmax:
        #     tgt_out = torch.diag(F.softmax(adv_pred, dim=-1)
        #                          [:, target_clone]).unsqueeze(-1)
        # else:
        #     tgt_out = torch.diag(adv_pred[:, target_clone]).unsqueeze(-1)
        # tgt_out.sum().backward()
        # grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(temp_grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads


class FGSMGradNAA:
    def __init__(self, epsilon, data_min, data_max,num_classes=1000):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.momentum = 1.0
        self.ens = 30.0
        self.gamma = 0.5
        self.num_classes = num_classes
        

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        find = False
        if model[1].__class__.__name__ == "Inception3":
            for name, module in model[1].named_modules():
                if "Mixed_5b" in name:
                    module.register_forward_hook(hook_feature)
                    find = True
                    break
        elif model[1].__class__.__name__ == "InceptionV4":
            for name, module in model[1].named_modules():
                if "features.5" in name:
                    module.register_forward_hook(hook_feature)
                    find = True
                    break
        elif model[1].__class__.__name__ == "ResNet":
            for name, module in model[1].named_modules():
                if "layer3.8" in name:
                    module.register_forward_hook(hook_feature)
                    find = True
                    break
        elif model[1].__class__.__name__ == "InceptionResNetV2":
            for name, module in model[1].named_modules():
                if "conv2d_4a.relu" in name:
                    module.register_forward_hook(hook_feature)
                    find = True
                    break
        if not find:
            for name, module in model[1].named_modules():
                if isinstance(module, nn.Conv2d):
                    module.register_forward_hook(hook_feature)
        alpha = self.epsilon / num_steps
        dt = data.clone().detach().requires_grad_(True)
        target = torch.nn.functional.one_hot(target, self.num_classes).to(device)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        grad_np = torch.zeros_like(dt)
        weight_np = None
        for step in range(num_steps):
            if step == 0:
                for l in range(int(self.ens)):
                    x_base = np.array([0.0, 0.0, 0.0])
                    x_base = image_transform(x_base)
                    images_base = image_transform(dt.clone())
                    images_base += (torch.randn_like(dt)*0.2 + 0)
                    images_base = images_base.cpu().detach().numpy().transpose(0, 2, 3, 1)
                    images_base = images_base * (1 - l / self.ens) + \
                        (l / self.ens) * x_base
                    images_base = torch.from_numpy(
                        images_base.transpose(0, 3, 1, 2)).float().to(device)

                    logits = model(images_base)
                    if use_softmax:
                        logits = F.softmax(logits, dim=-1)
                    if weight_np is None:
                        weight_np = torch.autograd.grad(
                            logits*target, features, grad_outputs=torch.ones_like(logits*target))[0]
                    else:
                        weight_np += torch.autograd.grad(
                            logits*target, features, grad_outputs=torch.ones_like(logits*target))[0]
                weight_np = -normalize(weight_np, 2)
            images_base = image_transform(torch.zeros_like(dt))
            _ = model(images_base)
            base_feamap = features
            _ = model(dt)
            adv_feamap = features
            loss = get_NAA_loss(adv_feamap, base_feamap, weight_np)
            grad = torch.autograd.grad(
                loss, dt)[0]
            grad = grad / torch.mean(torch.abs(grad), [1, 2, 3], keepdim=True)
            grad = self.momentum * grad_np + grad
            grad_np = grad.clone().detach().to(device)
            temp_output = model(dt)
            model.zero_grad()
            if use_softmax:
                temp_output = F.softmax(temp_output, dim=-1)
            loss = F.cross_entropy(temp_output, target.argmax(-1))
            loss.backward()
            temp_grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(temp_grad[i:i+1].clone())
            if use_sign:
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
                    grad_np = grad_np[keep_index, :]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        adv_pred = model(dt)
        adv_feamap = features
        loss = get_NAA_loss(adv_feamap, base_feamap, weight_np)
        grad = torch.autograd.grad(
            loss, dt)[0]
        grad = grad / torch.mean(torch.abs(grad), [1, 2, 3], keepdim=True)
        temp_output = model(dt)
        model.zero_grad()
        if use_softmax:
            temp_output = F.softmax(temp_output, dim=-1)
        loss = F.cross_entropy(temp_output, target_clone.argmax(-1))
        loss.backward()
        temp_grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(temp_grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone.argmax(-1)
        return dt, success, adv_pred, hats, grads




class FGSMGradSingle:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        alpha = self.epsilon / num_steps
        dt = data.clone().detach().requires_grad_(True)
        hats = [data.clone()]
        grads = list()
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            loss = self.criterion(output, target)
            loss.backward(retain_graph=True)
            if use_softmax:
                tgt_out = F.softmax(output, dim=-1)[:, target]
            else:
                tgt_out = output[:, target]
            grad = torch.autograd.grad(tgt_out, dt)[0]
            grads.append(grad.clone())
            if use_sign:
                data_grad = dt.grad.detach().sign()
                adv_data = dt + alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                hats.append(dt.data.clone())
            else:
                data_grad = grad / grad.norm()
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                hats.append(dt.data.clone())
            if early_stop:
                adv_pred = model(dt).argmax(-1)
                if adv_pred != target:
                    break
        adv_pred = model(dt)
        model.zero_grad()
        loss = self.criterion(adv_pred, target)
        loss.backward(retain_graph=True)
        if use_softmax:
            tgt_out = F.softmax(adv_pred, dim=-1)[:, target]
        else:
            tgt_out = adv_pred[:, target]
        grad = torch.autograd.grad(tgt_out, dt)[0]
        grads.append(grad.clone())
        hats = torch.cat(hats, dim=0)
        grads = torch.cat(grads, dim=0)
        success = adv_pred.argmax(-1) != target
        return dt, success, adv_pred, hats, grads


class MFABA:
    def __init__(self, model):
        self.model = model

    def __call__(self, hats, grads):
        t_list = hats[1:] - hats[:-1]
        grads = grads[:-1]
        total_grads = -torch.sum(t_list * grads, dim=0)
        attribution_map = total_grads.unsqueeze(0)
        return attribution_map.detach().cpu().numpy()


class MFABACOS:
    def __init__(self, model):
        self.model = model

    def __call__(self, data, baseline, hats, grads):
        input_clone = hats[0].clone().cpu().detach().numpy()
        baseline_clone = baseline.clone().cpu().detach().numpy()
        baseline_input = baseline_clone - input_clone
        t_list = list()
        for hat in hats:
            hat = hat.detach().cpu().numpy()
            hat_input = hat - input_clone
            t = np.sum(hat_input * baseline_input) / \
                (np.linalg.norm(baseline_input) ** 2)
            t_list.append(t)
        t_list = np.array(t_list)
        t_max = np.max(t_list)
        t_list = t_list / t_max

        n = len(grads)
        t_list = t_list[1:n] - t_list[0:n-1]
        total_grads = (grads[0:n-1] + grads[1:n]) / 2
        n = len(t_list)
        scaled_grads = total_grads.contiguous().view(
            n, -1) * torch.tensor(t_list).float().view(n, 1).to(grads.device)

        total_grads = torch.sum(scaled_grads.reshape(
            n, *data.shape[1:]), dim=0).unsqueeze(0)

        attribution_map = total_grads * (data - baseline)
        return attribution_map.detach().cpu().numpy()


class MFABANORM:
    def __init__(self, model):
        self.model = model

    def __call__(self, data, baseline, hats, grads):
        input_clone = hats[0].clone().cpu().detach().numpy()
        t_list = list()
        for hat in hats:
            hat = hat.detach().cpu().numpy()
            hat_input = hat - input_clone
            t = np.linalg.norm(hat_input)
            t_list.append(t)
        t_list = np.array(t_list)
        t_max = np.max(t_list)
        t_list = t_list / t_max
        n = len(grads)
        t_list = t_list[1:n] - t_list[0:n-1]
        total_grads = (grads[:n-1] + grads[1:]) / 2
        n = len(t_list)
        scaled_grads = total_grads.contiguous().view(
            n, -1) * torch.tensor(t_list).float().view(n, 1).to(grads.device)
        total_grads = torch.sum(scaled_grads.reshape(
            n, *data.shape[1:]), dim=0).unsqueeze(0)
        attribution_map = total_grads * (data - baseline)
        return attribution_map.detach().cpu().numpy()

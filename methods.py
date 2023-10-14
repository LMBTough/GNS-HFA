import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random
import scipy.stats as st
import copy
import math
from utils import ROOT_PATH
from functools import partial
import copy
import pickle as pkl
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Variable as V
from dct import *
from dataset import params
from model import get_model
from torchattacks import BIM as BIMO
from torchattacks import PGD as PGDO
from torchattacks import MIFGSM as MIFGSMO
from torchattacks import DIFGSM as DIFGSMO
from torchattacks import TIFGSM as TIFGSMO
from torchattacks import SINIFGSM as SINIFGSMO
from torchvision import transforms



def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + \
        (result > t_max).float() * t_max
    return result


def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(
        gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel


def DI(x, resize_rate=1.15, diversity_prob=0.5):
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


class BaseAttack(object):
    def __init__(self, attack_name, model_name, target):
        self.attack_name = attack_name
        self.model_name = model_name
        self.target = target
        if self.target:
            self.loss_flag = -1
        else:
            self.loss_flag = 1
        self.used_params = params(self.model_name)

        # loading model
        self.model = get_model(self.model_name)
        self.model.cuda()
        self.model.eval()

    def forward(self, *input):
        """
        Rewrite
        """
        raise NotImplementedError

    def _mul_std_add_mean(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        inps.mul_(std[:, None, None]).add_(mean[:, None, None])
        return inps

    def _sub_mean_div_std(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        # inps.sub_(mean[:,None,None]).div_(std[:,None,None])
        inps = (inps - mean[:, None, None])/std[:, None, None]
        return inps

    def _save_images(self, inps, filenames, output_dir):
        unnorm_inps = self._mul_std_add_mean(inps)
        for i, filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            image = unnorm_inps[i].permute([1, 2, 0])  # c,h,w to h,w,c
            image[image < 0] = 0
            image[image > 1] = 1
            image = Image.fromarray(
                (image.detach().cpu().numpy()*255).astype(np.uint8))
            # print ('Saving to ', save_path)
            image.save(save_path)

    def _update_inps(self, inps, grad, step_size):
        unnorm_inps = self._mul_std_add_mean(inps.clone().detach())
        unnorm_inps = unnorm_inps + step_size * grad.sign()
        unnorm_inps = torch.clamp(unnorm_inps, min=0, max=1).detach()
        adv_inps = self._sub_mean_div_std(unnorm_inps)
        return adv_inps

    def _update_perts(self, perts, grad, step_size):
        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts

    def _return_perts(self, clean_inps, inps):
        clean_unnorm = self._mul_std_add_mean(clean_inps.clone().detach())
        adv_unnorm = self._mul_std_add_mean(inps.clone().detach())
        return adv_unnorm - clean_unnorm

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images


class BIM(BaseAttack):
    def __init__(self, model_name):
        super(BIM, self).__init__('BIM', model_name, False)
        MEAN = self.used_params['mean']
        STD = self.used_params['std']
        norm_layer = transforms.Normalize(MEAN, STD)
        self.model = nn.Sequential(norm_layer, self.model).cuda().eval()
        self.bim = BIMO(self.model,eps=16/255, alpha=16/255/10, steps=10)
        
    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        adv_data = self.bim(unnorm_inps, labels).detach()
        return self._sub_mean_div_std(adv_data), None
        
class PGD(BaseAttack):
    def __init__(self, model_name):
        super(PGD, self).__init__('PGD', model_name, False)
        MEAN = self.used_params['mean']
        STD = self.used_params['std']
        norm_layer = transforms.Normalize(MEAN, STD)
        self.model = nn.Sequential(norm_layer, self.model).cuda().eval()
        self.pgd = PGDO(self.model,eps=16/255, alpha=16/255/10, steps=10)
        
    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        adv_data = self.pgd(unnorm_inps, labels).detach()
        return self._sub_mean_div_std(adv_data), None       

class MIFGSM(BaseAttack):
    def __init__(self, model_name):
        super(MIFGSM, self).__init__('MIFGSM', model_name, False)
        MEAN = self.used_params['mean']
        STD = self.used_params['std']
        norm_layer = transforms.Normalize(MEAN, STD)
        self.model = nn.Sequential(norm_layer, self.model).cuda().eval()
        self.mifgsm = MIFGSMO(self.model,eps=16/255, alpha=16/255/10, steps=10)
        
    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        adv_data = self.mifgsm(unnorm_inps, labels).detach()
        return self._sub_mean_div_std(adv_data), None   

class DIFGSM(BaseAttack):
    def __init__(self, model_name):
        super(DIFGSM, self).__init__('DIFGSM', model_name, False)
        MEAN = self.used_params['mean']
        STD = self.used_params['std']
        norm_layer = transforms.Normalize(MEAN, STD)
        self.model = nn.Sequential(norm_layer, self.model).cuda().eval()
        self.difgsm = DIFGSMO(self.model,eps=16/255, alpha=16/255/10, steps=10)
        
    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        adv_data = self.difgsm(unnorm_inps, labels).detach()
        return self._sub_mean_div_std(adv_data), None   

class TIFGSM(BaseAttack):
    def __init__(self, model_name):
        super(TIFGSM, self).__init__('TIFGSM', model_name, False)
        MEAN = self.used_params['mean']
        STD = self.used_params['std']
        norm_layer = transforms.Normalize(MEAN, STD)
        self.model = nn.Sequential(norm_layer, self.model).cuda().eval()
        self.tifgsm = TIFGSMO(self.model,eps=16/255, alpha=16/255/10, steps=10)
        
    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        adv_data = self.tifgsm(unnorm_inps, labels).detach()
        return self._sub_mean_div_std(adv_data), None   


class SINIFGSM(BaseAttack):
    def __init__(self, model_name):
        super(SINIFGSM, self).__init__('SINIFGSM', model_name, False)
        MEAN = self.used_params['mean']
        STD = self.used_params['std']
        norm_layer = transforms.Normalize(MEAN, STD)
        self.model = nn.Sequential(norm_layer, self.model).cuda().eval()
        self.sinifgsm = SINIFGSMO(self.model,eps=16/255, alpha=16/255/10, steps=10)
        
    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        adv_data = self.sinifgsm(unnorm_inps, labels).detach()
        return self._sub_mean_div_std(adv_data), None 


class TGR(BaseAttack):
    def __init__(self, model_name, sample_num_batches=130, steps=10, epsilon=16/255, target=False, decay=1.0):
        super(TGR, self).__init__('TGR', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.decay = decay

        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches
        # self._register_model()

    
    def _register_model(self):   
        def attn_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.model_name in ['vit_base_patch16_224', 'visformer_small', 'pit_b_224']:
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = min_all//H
                min_all_W = min_all%H
                out_grad[:,range(C),max_all_H,:] = 0.0
                out_grad[:,range(C),:,max_all_W] = 0.0
                out_grad[:,range(C),min_all_H,:] = 0.0
                out_grad[:,range(C),:,min_all_W] = 0.0
                
            if self.model_name in ['cait_s24_224']:
                B,H,W,C = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, H*W, C)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
                min_all_H = min_all//H
                min_all_W = min_all%H
                
                out_grad[:,max_all_H,:,range(C)] = 0.0
                out_grad[:,:,max_all_W,range(C)] = 0.0
                out_grad[:,min_all_H,:,range(C)] = 0.0
                out_grad[:,:,min_all_W,range(C)] = 0.0

            return (out_grad, )
        
        def attn_cait_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            
            B,H,W,C = grad_in[0].shape
            out_grad_cpu = out_grad.data.clone().cpu().numpy()
            max_all = np.argmax(out_grad_cpu[0,:,0,:], axis = 0)
            min_all = np.argmin(out_grad_cpu[0,:,0,:], axis = 0)
                
            out_grad[:,max_all,:,range(C)] = 0.0
            out_grad[:,min_all,:,range(C)] = 0.0
            return (out_grad, )
            
        def q_tgr(module, grad_in, grad_out, gamma):
            # cait Q only uses class token
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad[:] = 0.0
            return (out_grad, grad_in[1], grad_in[2])
            
        def v_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            if self.model_name in ['visformer_small']:
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = min_all//H
                min_all_W = min_all%H
                out_grad[:,range(C),max_all_H,max_all_W] = 0.0
                out_grad[:,range(C),min_all_H,min_all_W] = 0.0

            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
                    
                out_grad[:,max_all,range(c)] = 0.0
                out_grad[:,min_all,range(c)] = 0.0
            return (out_grad, grad_in[1])
        
        def mlp_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.model_name in ['visformer_small']:
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = min_all//H
                min_all_W = min_all%H
                out_grad[:,range(C),max_all_H,max_all_W] = 0.0
                out_grad[:,range(C),min_all_H,min_all_W] = 0.0
            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'resnetv2_101']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
        
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
                out_grad[:,max_all,range(c)] = 0.0
                out_grad[:,min_all,range(c)] = 0.0
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)
            return return_dics
                

        attn_tgr_hook = partial(attn_tgr, gamma=0.25)
        attn_cait_tgr_hook = partial(attn_cait_tgr, gamma=0.25)
        v_tgr_hook = partial(v_tgr, gamma=0.75)
        q_tgr_hook = partial(q_tgr, gamma=0.75)
        
        mlp_tgr_hook = partial(mlp_tgr, gamma=0.5)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_tgr_hook)
                self.model.blocks[i].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.blocks[block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.blocks[block_ind].mlp.register_backward_hook(mlp_tgr_hook)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(attn_cait_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.q.register_backward_hook(q_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.k.register_backward_hook(v_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.v.register_backward_hook(v_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.stage2[block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.stage2[block_ind].mlp.register_backward_hook(mlp_tgr_hook)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.stage3[block_ind-4].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.stage3[block_ind-4].mlp.register_backward_hook(mlp_tgr_hook)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(inps).cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()

        for i in range(self.steps):
            
            #add_perturbation = self._generate_samples_for_interactions(perts, i)
            #outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))

            ##### If you use patch out, please uncomment the previous two lines and comment the next line.

            outputs = self.model((self._sub_mean_div_std(unnorm_inps + perts)))
            cost = self.loss_flag * loss(outputs, labels).cuda()
            cost.backward()
            grad = perts.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=[1,2,3], keepdim=True)
            grad += momentum*self.decay
            momentum = grad
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None



class GNS_HFE(BaseAttack):
    def __init__(self, model_name, sample_num_batches=130, steps=10, epsilon=16/255, target=False, decay=1.0, ti=False, mi=False, scale=False, extreme=False,u=0.6,s=2,more_high_freq="",N=20):
        super(GNS_HFE, self).__init__('GNS_HFE', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.decay = decay

        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches
        self.T_kernel = gkern(7, 3)
        self.momentum = 1.0
        self.image_width = 224
        self.N = N
        # self.N = 10
        self.sigma = 16
        self.rho = 0.5
        self.ti = ti
        self.mi = mi
        self.scale = scale
        self.extreme = extreme
        self.more_high_freq = more_high_freq
        self.layer_grads = dict()
        self.layer_masks = dict()
        self.register_hooks = list()
        self.mask_hooks = list()
        self._register_model()
        self.u = u
        self.s = s

    def _register_model(self):
        def attn_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            # print(mask.shape)
            # print(grad_in[0].shape)
            # print(str(module))
            out_grad = mask * grad_in[0][:]
            return (out_grad, )

        def attn_cait_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            B, H, W, C = grad_in[0].shape
            return (out_grad, )

        def q_tgr(module, grad_in, grad_out, gamma):
            # cait Q only uses class token
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            return (out_grad, grad_in[1], grad_in[2])

        def v_tgr(module, grad_in, grad_out, name):
            if self.model_name in ['visformer_small']:
                B, C, H, W = grad_in[0].shape
                c_mus = torch.mean(abs(grad_in[0]), dim=[0, 2, 3]).cpu().numpy()
                mu = np.mean(c_mus)
                std = np.std(c_mus)
                muustd = mu + self.u * std
                c_factor = c_mus > muustd
                # c_temp = abs(np.tanh((c_mus-mu)/std))
                if self.s == 2:
                    c_temp = np.tanh(abs((c_mus-mu)/std))
                    c_factor = np.array(c_factor).astype(np.float32)
                    c_factor[c_factor == False] = c_temp[c_factor == False]
                elif self.s == 1:
                    c_temp = np.tanh(abs((c_mus-mu)/std))
                    c_factor = np.array(c_factor).astype(np.float32)
                    c_factor = c_temp
                elif self.s == 0:
                    c_temp = 1 - np.tanh(abs((c_mus-mu)/std))
                    c_factor = np.array(c_factor).astype(np.float32)
                    c_factor[c_factor == False] = c_temp[c_factor == False]
                # print(grad_in[0].shape)
                grad_in[0][:, :, :, :] = grad_in[0][:, :, :, :] * torch.from_numpy(c_factor).to(grad_in[0].device).view(1, C, 1, 1)
                
                                    
            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                c_mus = torch.mean(abs(grad_in[0]), dim=[0, 1]).cpu().numpy()
                mu = np.mean(c_mus)
                std = np.std(c_mus)
                muustd = mu + self.u * std
                # c_factor = [(c_mus[ix] > mu + self.u * std) for ix in range(c)]
                if self.s == 3:
                    c_factor = c_mus <= muustd
                else:
                    c_factor = c_mus > muustd
                grad_in[0][:, :, :] = grad_in[0][:, :, :] * torch.from_numpy(c_factor).float().to(grad_in[0].device).view(1, 1, c)

            mask = torch.ones_like(grad_in[0])
            out_grad = mask * grad_in[0][:]
            # self.layer_grads[name] = out_grad.cpu().numpy()
            return (out_grad, grad_in[1])

        def mlp_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)
            return return_dics

        attn_tgr_hook = partial(attn_tgr, gamma=1)
        attn_cait_tgr_hook = partial(attn_cait_tgr, gamma=1)
        v_tgr_hook = v_tgr
        q_tgr_hook = partial(q_tgr, gamma=1)
        mlp_tgr_hook = partial(mlp_tgr, gamma=1)

        if self.model_name in ['vit_base_patch16_224', 'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.register_hooks.append(self.model.blocks[i].attn.attn_drop.register_backward_hook(
                    attn_tgr_hook))
                
                self.register_hooks.append(self.model.blocks[i].attn.qkv.register_backward_hook(
                    partial(v_tgr_hook, name=f"model.blocks[{i}].attn.qkv")))
                self.register_hooks.append(self.model.blocks[i].mlp.register_backward_hook(mlp_tgr_hook))
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.register_hooks.append(self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(
                    attn_tgr_hook))
                self.register_hooks.append(self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(
                    partial(v_tgr_hook, name=f"model.transformers[{transformer_ind}].blocks[{used_block_ind}].attn.qkv")))
                self.register_hooks.append(self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(
                    mlp_tgr_hook))
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.register_hooks.append(self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(
                        attn_tgr_hook))
                    self.register_hooks.append(self.model.blocks[block_ind].attn.qkv.register_backward_hook(
                        partial(v_tgr_hook, name=f"model.blocks[{block_ind}].attn.qkv")))
                    self.register_hooks.append(self.model.blocks[block_ind].mlp.register_backward_hook(
                        mlp_tgr_hook))
                elif block_ind > 24:
                    self.register_hooks.append(self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(
                        attn_cait_tgr_hook))
                    self.register_hooks.append(self.model.blocks_token_only[block_ind -
                                                 24].attn.q.register_backward_hook(q_tgr_hook))
                    self.register_hooks.append(self.model.blocks_token_only[block_ind -
                                                    24].attn.k.register_backward_hook(partial(v_tgr_hook, name=f"model.blocks_token_only[{block_ind-24}].attn.k")))
                    self.register_hooks.append(self.model.blocks_token_only[block_ind -
                                                    24].attn.v.register_backward_hook(partial(v_tgr_hook, name=f"model.blocks_token_only[{block_ind-24}].attn.v")))
                    self.register_hooks.append(self.model.blocks_token_only[block_ind -
                                                 24].mlp.register_backward_hook(mlp_tgr_hook))
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.register_hooks.append(self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(
                        attn_tgr_hook))
                    self.register_hooks.append(self.model.stage2[block_ind].attn.qkv.register_backward_hook(
                        partial(v_tgr_hook, name=f"model.stage2[{block_ind}].attn.qkv")))
                    self.register_hooks.append(self.model.stage2[block_ind].mlp.register_backward_hook(
                        mlp_tgr_hook))
                elif block_ind >= 4:
                    self.register_hooks.append(self.model.stage3[block_ind -
                                      4].attn.attn_drop.register_backward_hook(attn_tgr_hook))
                    self.register_hooks.append(self.model.stage3[block_ind -
                                        4].attn.qkv.register_backward_hook(partial(v_tgr_hook, name=f"model.stage3[{block_ind-4}].attn.qkv")))
                    self.register_hooks.append(self.model.stage3[block_ind -
                                      4].mlp.register_backward_hook(mlp_tgr_hook))

    def free_register_hooks(self):
        for hook in self.register_hooks:
            hook.remove()
    
    def free_register_mask_hooks(self):
        for hook in self.mask_hooks:
            hook.remove()
    
    def generate_mask(self):
        grads = None
        for key in self.layer_grads:
            self.layer_masks[key] = torch.ones(self.layer_grads[key].shape).cuda()
            if grads is None:
                grads = self.layer_grads[key].reshape(-1)
            else:
                grads = np.concatenate((grads, self.layer_grads[key].reshape(-1)), axis=0)
        mu = np.mean(grads)
        sigma = np.var(grads)
        for key in self.layer_grads:
            mu_key = np.mean(self.layer_grads[key])
            self.layer_masks[key] = (1 + 0.6 * np.tanh((mu_key - mu) / sigma * 2)) * self.layer_masks[key]
    
                    
    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:, :, r*self.crop_length:(
                r+1)*self.crop_length, c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    
    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(inps).cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()

        for i in range(self.steps):

            # add_perturbation = self._generate_samples_for_interactions(perts, i)
            # outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))

            # If you use patch out, please uncomment the previous two lines and comment the next line.

            outputs = self.model((self._sub_mean_div_std(unnorm_inps + perts)))
            cost = self.loss_flag * loss(outputs, labels).cuda()
            cost.backward()
            grad = perts.grad.data
            grad = grad / torch.mean(torch.abs(grad),
                                     dim=[1, 2, 3], keepdim=True)
            grad += momentum*self.decay
            momentum = grad
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(
                unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None





class SSA(BaseAttack):
    def __init__(self, model_name, sample_num_batches=130, steps=10, epsilon=16/255, target=False, decay=1.0, ti=False, mi=False,more_high_freq=""):
        super(SSA, self).__init__('SSA', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.T_kernel = gkern(7, 3)
        self.momentum = 1.0
        self.image_width = 224
        self.N = 20
        self.sigma = 16
        self.rho = 0.5
        self.ti = ti
        self.mi = mi
        self.more_high_freq = more_high_freq

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        images_min = clip_by_tensor(unnorm_inps - self.epsilon, 0.0, 1.0)
        images_max = clip_by_tensor(unnorm_inps + self.epsilon, 0.0, 1.0)
        grad = 0
        x = unnorm_inps.clone().detach()
        x.requires_grad = True

        for i in range(self.steps):
            noise = 0
            for n in range(self.N):
                gauss = torch.randn(
                    x.size()[0], 3, self.image_width, self.image_width) * (self.sigma / 255)
                if self.more_high_freq == "noise" or self.more_high_freq == "both":
                    gauss = dct_2d(gauss)
                    msk = (np.ones((3,224,224)) * np.arange(224)[:,np.newaxis] * np.arange(224) / 223 / 223)
                    msk = torch.from_numpy(msk).float()
                    gauss = gauss * msk
                    gauss = idct_2d(gauss)

                gauss = gauss.cuda()
                x_dct = dct_2d(x + gauss).cuda()
                if self.more_high_freq == "search" or self.more_high_freq == "both":
                    msk = (np.ones((3,224,224)) * np.linspace(112,224,224)[:,np.newaxis] * np.linspace(112,224,224) / 224 / 224)
                    msk = torch.from_numpy(msk).float().cuda()
                    mask = (torch.rand_like(x) * msk * 2 * self.rho + 1 - self.rho).cuda()
                else:
                    mask = (torch.rand_like(x) * 2 * self.rho +
                            1 - self.rho).cuda()  # 高频多探索低频少探索
                x_idct = idct_2d(x_dct * mask)
                x_idct = V(x_idct, requires_grad=True)

                # DI-FGSM https://arxiv.org/abs/1803.06978
                # output_v3 = model(DI(x_idct))

                output_v3 = self.model(self._sub_mean_div_std(x_idct))
                loss = F.cross_entropy(output_v3, labels)
                loss.backward()
                noise += x_idct.grad.data
            noise = noise / self.N
            if self.ti:
                noise = F.conv2d(noise, self.T_kernel, bias=None,
                                stride=1, padding=(3, 3), groups=3)

            # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
            if self.mi:
                noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
                noise = self.momentum * grad + noise
                grad = noise
            x = x + self.step_size * torch.sign(noise)
            x = clip_by_tensor(x, images_min, images_max)
        return self._sub_mean_div_std(x).detach(), None

    
class PNA(BaseAttack):
    def __init__(self, model_name, ablation_study='1,1,1', sample_num_batches=130, lamb=0.1, steps=10, epsilon=16/255, target=False):
        super(PNA, self).__init__('PNA', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps

        self.ablation_study = ablation_study.split(',')
        self.lamb = lamb
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches

        if self.ablation_study[2] == '1':
            print ('Using Skip')
            self._register_model()
        else:
            print ('Not Using Skip')
    
    def _register_model(self):   
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )

        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
                for i in range(12):
                    self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()

        for i in range(self.steps):
            if self.ablation_study[0] == '1':
                add_perturbation = self._generate_samples_for_interactions(perts, i)
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))
            else:
                outputs = self.model((self._sub_mean_div_std(unnorm_inps + perts)))

            if self.ablation_study[1] == '1':
                cost1 = self.loss_flag * loss(outputs, labels).cuda()
                cost2 = torch.norm(perts)
                cost = cost1 + self.lamb * cost2
            else:
                cost = self.loss_flag * loss(outputs, labels).cuda()
            cost.backward()
            grad = perts.grad.data
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None

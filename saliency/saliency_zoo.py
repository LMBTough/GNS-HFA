from saliency.core import pgd_step,pgd_ssa_step,DL, BIG,SSA, FGSM, MFABA, MFABACOS, MFABANORM, FGSMGradSingle, FGSMGrad, IntegratedGradient, SaliencyGradient, SmoothGradient,FGSMGradSSA
from saliency.core import PGDGrad,DIFGSMGrad,TIFGSMGrad,MIFGSMGrad,SINIFGSMGrad,FGSMGradNAA
from saliency.core import DIFGSMGrad_ori,TIFGSMGrad_ori,MIFGSMGrad_ori
from saliency.core import FastIG, GuidedIG,SaliencyMap,AttributionPriorExplainer
import torch
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def agi(model, data, target, epsilon=0.05, max_iter=20, topk=20):
    model = model[:2]
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = random.sample(list(range(0, 999)), topk)
    output = model(data)

    init_pred = output.argmax(-1)

    top_ids = selected_ids

    step_grad = 0

    for l in top_ids:

        targeted = torch.tensor([l] * data.shape[0]).to(device)

        if l < 999:
            targeted[targeted == init_pred] = l + 1
        else:
            targeted[targeted == init_pred] = l - 1

        delta, perturbed_image = pgd_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy()
    return adv_ex


def agi_ssa(model, data, target, epsilon=0.05, max_iter=20, topk=1):
    model = model[:2]
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = random.sample(list(range(0, 999)), topk)
    output = model(data)

    init_pred = output.argmax(-1)

    top_ids = selected_ids

    step_grad = 0

    for l in top_ids:

        targeted = torch.tensor([l] * data.shape[0]).to(device)

        if l < 999:
            targeted[targeted == init_pred] = l + 1
        else:
            targeted[targeted == init_pred] = l - 1

        delta, perturbed_image = pgd_ssa_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy()
    return adv_ex


def big(model, data, target, data_min=0, data_max=1, epsilons=[16], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map

def big_ssa(model, data, target, data_min=0, data_max=1, epsilons=[16], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [SSA(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map


def mfaba_smooth(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    epsilon = epsilon / 255
    mfaba = MFABA(model)
    attack = FGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def mfaba_ssa_smooth(model, data, target, data_min=0, data_max=1, epsilon=16,N=20,num_steps=10, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    epsilon = epsilon / 255
    attack = FGSMGradSSA(
        epsilon=epsilon, data_min=data_min, data_max=data_max,N=N)
    _, _, _, hats, grads = attack(
        model, data, target,num_steps=num_steps, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map


def mfaba_pgd_smooth(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    epsilon = epsilon / 255
    attack = PGDGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def mfaba_difgsm_smooth(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    epsilon = epsilon / 255
    attack = DIFGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def mfaba_tifgsm_smooth(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    epsilon = epsilon / 255
    attack = TIFGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map


def mfaba_mifgsm_smooth(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    epsilon = epsilon / 255
    attack = MIFGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map




def mfaba_difgsmori_smooth(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    epsilon = epsilon / 255
    attack = DIFGSMGrad_ori(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def mfaba_tifgsmori_smooth(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    epsilon = epsilon / 255
    attack = TIFGSMGrad_ori(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map


def mfaba_mifgsmori_smooth(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    epsilon = epsilon / 255
    attack = MIFGSMGrad_ori(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map


def mfaba_sinifgsm_smooth(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    epsilon = epsilon / 255
    attack = SINIFGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def mfaba_naa_smooth(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    epsilon = epsilon / 255
    attack = FGSMGradNAA(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def mfaba_sharp(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=False, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    epsilon = epsilon / 255
    mfaba = MFABA(model)
    attack = FGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map


def mfaba_cos(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=False, use_softmax=False):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    epsilon = epsilon / 255
    mfaba_cos = MFABACOS(model)
    attack = FGSMGradSingle(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = mfaba_cos(data, dt, hats, grads)
    return attribution_map


def mfaba_norm(model, data, target, data_min=0, data_max=1, epsilon=16, use_sign=False, use_softmax=False):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    epsilon = epsilon / 255
    mfaba_norm = MFABANORM(model)
    attack = FGSMGradSingle(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    dt, success, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = mfaba_norm(data, dt, hats, grads)
    return attribution_map


def ig(model, data, target, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ig = IntegratedGradient(model)
    return ig(data, target, gradient_steps=gradient_steps)


def sm(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sm = SaliencyGradient(model)
    return sm(data, target)


def sg(model, data, target, stdevs=0.15, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sg = SmoothGradient(model, stdevs=stdevs)
    return sg(data, target, gradient_steps=gradient_steps)


def deeplift(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    dl = DL(model)
    return dl(data, target)

import copy
def fast_ig(model, data, target, *args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    method = FastIG(model)
    result = method(data, target).squeeze()
    return np.expand_dims(result, axis=0)

def guided_ig(model, data, target, steps=15):
    model = copy.deepcopy(model)[:2]
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    class_idx_str = 'class_idx_str'

    def call_model_function(images, call_model_args=None, expected_keys=None):
        target_class_idx = call_model_args[class_idx_str]
        images = torch.from_numpy(images).float().to(device)
        images = images.requires_grad_(True)
        output = model(images)
        # m = torch.nn.Softmax(dim=1)
        # output = m(output)
        output = output
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(
            outputs, images, grad_outputs=torch.ones_like(outputs))[0]
        gradients = grads.cpu().detach().numpy()
        return {'INPUT_OUTPUT_GRADIENTS': gradients}

    im = data.squeeze().cpu().detach().numpy()
    call_model_args = {class_idx_str: target}
    baseline = np.zeros(im.shape)
    method = GuidedIG()

    result = method.GetMask(
        im, call_model_function, call_model_args, x_steps=steps, x_baseline=baseline)
    return np.expand_dims(result, axis=0)

def saliencymap(model, data, target, *args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    saliencymap = SaliencyMap(model)
    return saliencymap(data, target)


def eg(model,dataloader,data,target,*args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    APExp = AttributionPriorExplainer(dataloader.dataset, 4,k=1)
    attr_eg = APExp.shap_values(model,data).cpu().detach().numpy()
    return attr_eg
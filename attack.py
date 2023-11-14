import torch
import argparse
from torch.utils.data import DataLoader
import os
import pandas as pd
import time
import pickle as pkl
from tqdm import tqdm
from dataset import AdvDataset
from utils import BASE_ADV_PATH, ROOT_PATH
import methods
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
setup_seed(2023)

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--attack', type=str, default='', help='the name of specific attack method')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='input batch size for reference (default: 16)')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--filename_prefix', type=str, default='', help='')
    parser.add_argument('--ti', action='store_true', default=False, help='whether to use TI')
    parser.add_argument('--mi', action='store_true', default=False, help='whether to use MI')
    parser.add_argument('--mhf', default="",choices=["","noise","search","both"], help='whether to add more high freq')
    parser.add_argument('--u', type=float, default=1)    
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--eps", type=int, default=16)
    
    args = parser.parse_args()
    args.opt_path = os.path.join(BASE_ADV_PATH,f"model_{args.model_name}-method_{args.attack}-ti_{args.ti}-mi_{args.mi}-mhf_{args.mhf}-u_{args.u}-N_{args.N}-eps_{args.eps}")
    if not os.path.exists(args.opt_path):
        os.makedirs(args.opt_path)
    return args

if __name__ == '__main__':
    args = arg_parse()
    # loading dataset
    dataset = AdvDataset(args.model_name, os.path.join(ROOT_PATH, 'clean_resized_images'))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print (args.attack, args.model_name)
    
    # Attack
    if args.attack == "TGR":
        attack_method = getattr(methods, args.attack)(args.model_name)
    elif args.attack == "SSA":
        attack_method = getattr(methods, args.attack)(args.model_name,ti=args.ti,mi=args.mi,more_high_freq=args.mhf)
    elif args.attack == "GNS_HFE":
        attack_method = getattr(methods, args.attack)(args.model_name,ti=args.ti,mi=args.mi,more_high_freq=args.mhf,u=args.u,N=args.N,epsilon=args.eps/255)
    else:
        attack_method = getattr(methods, args.attack)(args.model_name)
    pbar = tqdm(total=len(data_loader))
    # Main
    all_loss_info = {}
    for batch_idx, batch_data in enumerate(data_loader):
        batch_x = batch_data[0]
        batch_y = batch_data[1]
        batch_name = batch_data[3]
        adv_inps, loss_info = attack_method(batch_x, batch_y)
        attack_method._save_images(adv_inps, batch_name, args.opt_path)
        pbar.update(1)

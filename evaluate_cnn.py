import os
import torch
import csv
from torch.autograd import Variable as V
from torch import nn
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from torch.utils.data import DataLoader
from utils import BASE_ADV_PATH, accuracy, AverageMeter
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
from dataset import CNNDataset
import argparse

import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# setup_seed(2023)

batch_size = 10

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--adv_path', type=str, default='', help='the path of adversarial examples.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='input batch size for reference (default: 16)')
    parser.add_argument('--model_name', type=str, default='', help='')
    args = parser.parse_args()
    args.adv_path = os.path.join(BASE_ADV_PATH, args.adv_path)
    return args

args = arg_parse()
adv_dir = args.adv_path
batch_size = args.batch_size
# adv_dir = './advimages/model_vit_base_patch16_224-method_TGR'
# adv_dir = './advimages/model_visformer_small-method_TGR'
# adv_dir = './advimages/model_pit_b_224-method_TGR'
# adv_dir = './advimages/model_cait_s24_224-method_TGR'


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf2torch_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf2torch_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf2torch_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf2torch_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf2torch_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf2torch_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf2torch_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        net.KitModel(model_path).eval().cuda(),)
    return model

def verify(model_name, path):

    model = get_model(model_name, path)

    dataset = CNNDataset("inc-v3", adv_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sum = 0
    for batch_idx, batch_data in enumerate(data_loader):
        batch_x = batch_data[0].cuda()
        batch_y = batch_data[1].cuda()
        batch_name = batch_data[2]

        with torch.no_grad():
            sum += (model(batch_x)[0].argmax(1) != batch_y+1).detach().sum().cpu()

    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))
    asr = sum / 1000.0
    return model_name,asr

def main():
    model_names = ['tf2torch_inception_v3','tf2torch_inception_v4','tf2torch_inc_res_v2','tf2torch_resnet_v2_101','tf2torch_ens3_adv_inc_v3','tf2torch_ens4_adv_inc_v3','tf2torch_ens_adv_inc_res_v2']

    models_path = './models/'
    csv_filename = f"outputs/{os.path.basename(args.adv_path)}_output_cnn.csv"
    # if os.path.exists(csv_filename):
    #     print("CSV 文件已存在。")
    #     os.remove(csv_filename)
    for model_name in model_names:
        model_name_csv,asr = verify(model_name, models_path)
        print("===================================================")
        # 创建一个列表，用于存储要写入 CSV 表格的数据
        data_to_write = []

        # 添加 adv_path、model_name 和 success_count/1000. * 100 到数据列表中
        data_to_write.append({
            'adv_path': adv_dir,
            'model_name': model_name_csv,
            'ASR': asr.item()
        })

        # 指定要写入的文件名
        csv_filename = f"outputs/{os.path.basename(args.adv_path)}_output_cnn.csv"
        
        # 将数据写入 CSV 表格
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['adv_path', 'model_name', 'ASR']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_to_write)

        print("CSV 文件已成功写入。")
        

if __name__ == '__main__':
    main()
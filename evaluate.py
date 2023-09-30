import torch
import argparse
from torch.utils.data import DataLoader
import os
import pandas as pd
import time
import csv
from dataset import AdvDataset
from model import get_model
from utils import BASE_ADV_PATH, accuracy, AverageMeter

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

if __name__ == '__main__':
    args = arg_parse()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Loading dataset
    dataset = AdvDataset(args.model_name, args.adv_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print (len(dataset))
    # Loading model
    model = get_model(args.model_name)
    model.cuda()
    model.eval()

    # main 
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    prediction = []
    gts = []
    with torch.no_grad():
        end = time.time()
        for batch_idx, batch_data in enumerate(data_loader):
            #if batch_idx%10 == 0:
            #    print ('Ruing batch_idx', batch_idx)
            batch_x = batch_data[0].cuda()
            batch_y = batch_data[1].cuda()
            batch_name = batch_data[2]

            output = model(batch_x)
            acc1, acc5 = accuracy(output.detach(), batch_y, topk=(1, 5))
            top1.update(acc1.item(), batch_x.size(0))
            top5.update(acc5.item(), batch_x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = output.detach().topk(1, 1, True, True)
            pred = pred.t()
            prediction += list(torch.squeeze(pred.cpu()).numpy())
            gts += list(batch_y.cpu().numpy())
    success_count = 0
    df = pd.DataFrame(columns = ['path', 'pre', 'gt'])
    df['path'] = dataset.paths[:len(prediction)]
    df['pre'] = prediction
    df['gt'] = gts
    for i in range(len(df['pre'])):
        if df['pre'][i] != df['gt'][i]:
            success_count += 1
    print("Attack Success Rate for {0} : {1:.1f}%".format(args.model_name, success_count/ len(dataset) * 100))
    # 创建一个列表，用于存储要写入 CSV 表格的数据
    data_to_write = []

    # 添加 adv_path、model_name 和 success_count/len(dataset). * 100 到数据列表中
    data_to_write.append({
        'adv_path': args.adv_path,
        'model_name': args.model_name,
        'ASR': success_count / len(dataset) * 100
    })

    # 指定要写入的文件名
    # csv_filename = 'output.csv'
    if not os.path.exists('outputs'):
        os.makedirs('outputs',exist_ok=True)
    # if os.path.exists(f"outputs/{os.path.basename(args.adv_path)}_output.csv") and args.model_name == 'levit_256':
    #     os.remove(f"outputs/{os.path.basename(args.adv_path)}_output.csv")
    csv_filename = f"outputs/{os.path.basename(args.adv_path)}_output.csv"

    # 将数据写入 CSV 表格
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['adv_path', 'model_name', 'ASR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_to_write)

    print("CSV 文件已成功写入。")

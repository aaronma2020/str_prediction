import argparse
import os
import torch
import random
import numpy as np
import pickle
from tqdm import tqdm
from utils import set_seed, compute_smi, save_log
from data_load.load import load_data
from models.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    # 设置随机种子
    set_seed(1)

    # 读取质谱数据
    train_data, val_data = load_data(args.ms_data, args.group, args.batch_size, args.accuracy)
    with open(args.for2smiles, 'rb') as f:
        for2smiles = pickle.load(f)
    # 模型
    model = Model(args.accuracy).to(device)
    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    step = len(train_data)
    for epoch in range(1,args.epoch+1):
        print(f'*** 开始训练第{epoch}轮 ***')
        epoch_loss = 0
        model.train()
        for i, (mz_vec, fid, target, formula, smiles) in tqdm(enumerate(train_data)):
            mz_vec = mz_vec.to(device)
            fid = fid.to(device)
            target = target.squeeze(1).to(device)

            prob = model(mz_vec, fid)

            loss = loss_fn(prob, target)
            epoch_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = round(epoch_loss/step, 4)
        print(epoch_loss)

        # 测试
        if epoch % 1 == 0:
            print(f'*** 开始验证第{epoch}轮 ***')
            model.eval()
            top1_sum = 0
            top5_sum = 0
            top10_sum = 0
            num = len(val_data)

            for i, (mz_vec, fid, target, formula, smiles) in tqdm(enumerate(val_data)):

                with open(os.path.join(args.fingerid, formula[0] + '.pkl'), 'rb') as f:
                    fid_list = pickle.load(f)
                fid = torch.Tensor(fid_list).to(device)
                can_num = len(fid)
                _, dim = mz_vec.shape
                mz_vec = mz_vec.expand(*[can_num, dim]).to(device)

                prob = torch.softmax(model(mz_vec, fid), 1)[:,1]

                can_smiles = for2smiles[formula[0]]
                # 计算相似性
                top1, top5, top10 = compute_smi(prob, smiles[0], can_smiles)
                top1_sum += top1
                top5_sum += top5
                top10_sum += top10


            top1_acc = top1_sum/num
            top5_acc = top5_sum/num
            top10_acc = top10_sum/num

            save_log(epoch, epoch_loss, top1_acc, top5_acc, top10_acc, args.log)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 读取参数
    parser.add_argument('--ms_data', type=str, default='./data/process_data/ms_data.pkl', help='ms数据')
    parser.add_argument('--group', type=int, default=0, help='交叉验证数据分组')
    parser.add_argument('--for2smiles', type=str, default='./data/process_data/for2smiles.pkl', help='查询分子式的异构体smiles')
    parser.add_argument('--fingerid', type=str, default='./data/process_data/fingerid', help='异构体的fingerid' )
    # 模型参数
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--accuracy', type=int, default=10000, help='质普数据精度')
    parser.add_argument('--lr', type=float, default=1e-4, help='模型学习率')


    # data loader 参数
    parser.add_argument('--batch_size',type=int, default=100)

    # 保存参数
    parser.add_argument('--log', type=str, default='./log.csv')

    args = parser.parse_args()
    main(args)
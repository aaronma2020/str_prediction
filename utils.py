import torch
import random
import pandas as pd
import numpy as np
import os
import pickle
import re

def set_seed(seed):
    ''' 固定随机种子'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


atom2mass = {
    'H': 1.007825,
    'B': 10.0129,
    'C': 12.0,
    'N': 14.00307,
    'O': 15.99491,
    'F': 18.99840,
    'Si': 27.9769,
    'P': 30.97376,
    'S': 31.9721,
    'Cl': 34.9689,
    'Br': 78.9183,
    'I': 126.90448,
}


def extra_atoms(formula):
    '''提取分子式各个原子的个数'''

    atom_dict = {}
    pattern = re.compile(r'[A-Z][a-z]?[0-9]*')
    items = pattern.findall(formula)
    for item in items:
        pattern = re.compile(r'[A-Z][a-z]?')
        atom = pattern.findall(item)[0]
        pattern = re.compile(r'[0-9]+')
        num = pattern.findall(item)
        if len(num) != 0:
            num = num[0]
        else:
            num = 1

        atom_dict[atom] = num
    return atom_dict

def nsum(frament_dict, atom2mass):
    '''计算部分碎片对应的质量数'''

    atoms = frament_dict.keys()
    mass = 0
    for atom in atoms:
        mass += atom2mass[atom] * int(frament_dict[atom])
    return mass

def remove(mz, intensity, formula):
    '''去掉大于分子质量的杂质碎片,并取前top个'''
    formula_dict = extra_atoms(formula)
    formula_mass = nsum(formula_dict, atom2mass)
    mz = np.array(list(map(float, mz)))
    intensity = np.array(list(map(float, intensity)))
    try:
        index = np.where(mz >= formula_mass)[0][0]
    except:
        index = len(mz)
    mz = mz[:index + 1]
    intensity = intensity[:index + 1]
    front_indexs = np.argsort(intensity)[::-1]
    new_intensity = [intensity[i] for i in front_indexs]
    new_mz = [mz[i] for i in front_indexs]

    return new_mz, new_intensity

def com_frament(new_dict):
    '''把找到的碎片计算出来分子式'''
    result = []
    for key, value in new_dict.items():
        if value != 0:
            result.append(key + str(value))
    return ''.join(result)


def carry(new_dict, atoms_dict):
    tag = False
    atoms = list(new_dict.keys())
    num = len(new_dict)
    for i, (key, value) in enumerate(new_dict.items()):
        if i == 0:
            new_dict[key] = 0
            new_dict[atoms[i + 1]] += 1
        elif value > int(atoms_dict[key]) and i + 1 < num:
            new_dict[key] = 0
            new_dict[atoms[i + 1]] += 1
        if new_dict[atoms[num - 1]] == int(atoms_dict[atoms[num - 1]]):
            tag = True

    return new_dict, tag

def compute_smi(prob, smiles, can_smiles):
    '''计算相似性,返回前10'''
    prob = prob.to('cpu')
    indexs = list(map(int, torch.argsort(prob,dim=0)))[::-1]
    # 计算top1,top5,top10正确率
    top1 = 0
    top5 = 0
    top10 = 0
    top10_candidate = []
    top5_candidate = []

    if smiles == can_smiles[indexs[0]]:
        top1 = 1
    if len(indexs) <=5:
        top5 = 1
        top10 = 1
    else:
        for i, index in enumerate(indexs):
            if i == 5:
                break
            top5_candidate.append(can_smiles[index])
        if smiles in top5_candidate:
            top5 = 1
        if len(indexs) <=10:
            top10 = 1
        else:
            for i, index in enumerate(indexs):
                if i == 10:
                    break
                top10_candidate.append(can_smiles[index])
            if smiles in top10_candidate:
                top10 = 1

    return top1, top5, top10


def save_log(epoch, loss, top1_acc, top5_acc, top10_acc, save_path):
    '''保存训练信息'''

    if epoch == 10:
        log_list = ['epoch', 'loss', 'top1_acc', 'top5_acc', 'top10_acc']
        log_file = pd.DataFrame(columns=log_list)
        log_file.to_csv(save_path, index=0)

    log_file = pd.read_csv(save_path)
    score_dict = {'epoch': epoch, 'top1_acc': top1_acc, 'top5_acc': top5_acc, 'top10_acc':top10_acc}
    log_file = log_file.append(score_dict, ignore_index=True)
    log_file.to_csv(save_path, index=0)
    print(f'loss:{loss}, top1:{top1_acc}, top5:{top5_acc}, top10:{top10_acc}')











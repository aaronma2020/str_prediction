'''

'''
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import re
import pickle
from rdkit import Chem

class Dataset(data.Dataset):

    def __init__(self, data_path, group, type, accuracy=10000):
        with open(data_path,'rb') as f:
            data = pickle.load(f)
        self.acc = accuracy
        if type == 'train':
            self.data = data[data.group!=group]
        else:
            self.data = data[(data.group==group) & (data.target==1)]
        with open('./data/process_data/ms_data_remove1.pkl', 'rb') as f:
            self.remove_ms = pickle.load(f)
    def __getitem__(self, index):

        item = self.data.iloc[index]
        formula = item['formula']
        smiles = item['smiles']
        csv_name = item['csv_name']
        mz = self.remove_ms[csv_name][0]
        intensity = self.remove_ms[csv_name][1]
        frament = self.remove_ms[csv_name][1]
        target = torch.Tensor([item['target']]).long()

        sum_int = sum(intensity)
        mz_vec = torch.zeros(self.acc)
        for i,m in enumerate(mz):
            if m >= 1000:
                continue
            index = int(m*(self.acc / 1000))
            mz_vec[index] = intensity[i] / sum_int
        mol = Chem.MolFromSmiles(smiles)
        fid= torch.Tensor(list(Chem.RDKFingerprint(mol)))


        return mz_vec, fid, target, formula, smiles

    def __len__(self):
        return len(self.data)

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
    '''去掉大于分子质量的杂质碎片'''
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

def load_data(data_path, group, batch_size, accuracy=10000):

    trainset = Dataset(data_path, group, 'train', accuracy=accuracy)
    valset = Dataset(data_path, group, 'val', accuracy=accuracy)
    train_loader = data.DataLoader(dataset=trainset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)

    val_loader = data.DataLoader(dataset=valset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=8,
                                   drop_last=True)

    return train_loader, val_loader
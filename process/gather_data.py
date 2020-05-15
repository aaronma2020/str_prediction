'''
把所有的质谱信息都保存在一个csv里
'''
import pickle
import random
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from rdkit import Chem
import sys
sys.path.append('..')
from utils import set_seed

set_seed(1)

neg_fold = 10   #正负样本比例

id_sdf = '../data/original_data/sdf'
csv_dir = '../data/original_data/spectral'
group_path = '../data/original_data/group_inf.csv'
id_path = '../data/original_data/GPNS_inf.csv'
for2smi_path = '../data/process_data/for2smiles.pkl'
csv_list = os.listdir(csv_dir)

group_csv = pd.read_csv(group_path, names=['file','group'])
id_csv = pd.read_csv(id_path, names=['file','formula','id'])

gather_mz = pd.DataFrame(columns=['id', 'formula','mz', 'intensity','smiles', 'group'])

# 正样本
for csv_f in tqdm(csv_list):
    path = os.path.join(csv_dir,csv_f)
    csv = pd.read_csv(path, names=['id', 'formula','mz', 'intensity'])

    #获取对应的mz,intensity
    mz = csv['mz'].to_list()
    intensity = csv['intensity'].to_list()

    csv_name = csv_f[:-4]

    #查找对应的cid和formula
    index = id_csv[id_csv.file == csv_name].index.tolist()[0]
    id = id_csv.iloc[index]['id']
    formula = id_csv.iloc[index]['formula']
    #查找交叉验证分组信息

    index = group_csv[group_csv.file==csv_name].index.tolist()[0]
    group = group_csv.iloc[index]['group']
    #查找对应的smiles式
    path = os.path.join(id_sdf,str(id)+'.sdf')
    mols = Chem.SDMolSupplier(path)
    for mol in mols:
        smi = Chem.MolToSmiles(mol)

    one_item = {'id':id, 'formula':formula, 'mz':mz, 'intensity':intensity, 'smiles':smi, 'group':group, 'csv_name':csv_name, 'target':1}
    gather_mz = gather_mz.append(one_item, ignore_index=True)

# 负样本
with open(for2smi_path, 'rb') as f:
    for2smi = pickle.load(f)
for index in tqdm(gather_mz.index):
    item = gather_mz.iloc[index]
    id = item['id']
    formula = item['formula']
    mz = item['mz']
    intensity = item['intensity']
    pos_smi = item['smiles']
    group = item['group']
    csv_nam = item['csv_name']
    num = len(for2smi[formula])
    if num == 1:
        continue
    else:
        for i in range(neg_fold):
            while(1):
                ran_num = random.randint(0, num-1)
                neg_smi = for2smi[formula][ran_num]
                if neg_smi != pos_smi:
                    break
            one_item =  one_item = {'id':id, 'formula':formula, 'mz':mz, 'intensity':intensity, 'smiles':neg_smi, 'group':group, 'csv_name':csv_name, 'target':0}
            gather_mz = gather_mz.append(one_item, ignore_index=True)
with open(f'../data/process_data/ms_data_1to{neg_fold}.pkl', 'wb') as f:
    pickle.dump(gather_mz, f)






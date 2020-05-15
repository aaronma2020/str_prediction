'''
通过rdKit提取候选物质的fid
生成一个文件夹fid，里面按照分子式的名字保存([formula].pkl),每个pkl里面是一个list，具体格式为[[fingerid1], [fingerid2]]
同时生成一个for2smiles.pkl文件，用于查询每个分子式的同分异构体的smiles，具体格式为{'formula1':[smiles1, smiles2, ...],'formula2':[smiles1, smiles2, ...]}
'''
import os
from rdkit import Chem
import pickle
from tqdm import tqdm

in_dir = '../data/original_data/formula_sdf'
fid_dir = '../data/process_data/fid'
for2smiles_path = '../data/process_data/for2smiles.pkl'
formula_sdf = os.listdir(in_dir)
bad_smiles = 0
for2smiles = {}  # [[smiles, atoms, edge, adge_att ], [],...]
for formula in tqdm(formula_sdf):
    path = os.path.join(in_dir, formula)
    mols = Chem.SDMolSupplier(path)
    smi_list = []
    fid_list = []
    for mol in mols:
        # 提取smiles
        try:
            smi = Chem.MolToSmiles(mol)
            fid = list(Chem.RDKFingerprint(mol))
            fid_list.append(fid)
            smi_list.append(smi)
        except:
            bad_smiles += 1

    for2smiles[formula[:-4]] = smi_list
    with open(fid_dir + formula[:-3] + 'pkl', 'wb') as f:
        pickle.dump(fid_list, f)

with open(for2smiles_path, 'wb') as f:
    pickle.dump(for2smiles, f)
print(f'*** 无法识别的 smiles:{bad_smiles}')
'''从sdf文件中提取数据
'''
import argparse
import os
from rdkit import Chem
import pickle
from tqdm import tqdm

in_dir = '../data/original/formula_sdf'
out_dir = '../data/process_data/graph_pkl'

formula_sdf = os.listdir(in_dir)

total_pkl = { } # [[smiles, atoms, edge, adge_att ], [],...]
for formula in tqdm(formula_sdf):
    path = os.path.join(in_dir, formula)
    mols = Chem.SDMolSupplier(path)
    for mol in mols:
        mol_pkl = []

        # 提取smiles
        smi = Chem.MolToSmiles(mol)
        mol_pkl.append(smi)
        atoms = mol.GetAtoms()

        # 提取原子信息
        atom_list = []
        for atom in atoms:
            atom_list.append(atom.GetSymbol())
        mol_pkl.append(atom_list)
        total_pkl.append(mol_pkl)

        # 提取边的信息

        edges = mol.GetBonds()
        edge_list = [[],[]]
        edge_att = []
        for e in edges:

            edge_att.append(str(e.GetBondType()))
            edge_list[0].extend([e.GetBeginAtomIdx(), e.GetEndAtomIdx()])
            edge_list[1].extend([ e.GetEndAtomIdx(), e.GetBeginAtomIdx()])

        total_pkl.append(edge_list)
        total_pkl.append(edge_att)

with open(args.out_file, 'wb') as f:
    pickle.dump(total_pkl, f)



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='../data/formula_sdf', help='分子式的sdf文件夹')
    parser.add_argument('--out_file', type=str, default='../data/total_pkl', help='生成需要的信息的pkl文件夹')

    args = parser.parse_args()
    main(args)




import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class Model(nn.Module):

    def __init__(self, accuracy):

        super(Model,self).__init__()

        self.spec_nn = nn.Linear(10000,2048)
        self.fid_nn = nn.Linear(2048, 128)
        self.fc =  nn.Linear(2048, 2)
        self.dropout = nn.Dropout(0.2)
    def forward(self, mz_vec, fid):

        spec_fea = self.spec_nn(mz_vec)
        fid_fea = fid
        fusion = spec_fea+fid_fea
        prob = self.fc(self.dropout(fusion))
        return prob





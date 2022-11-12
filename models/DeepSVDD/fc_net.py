import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, input_size, out_size, dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(input_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class FC_net(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.rep_dim = 128
        self.block1 = Block(in_dim, 512)
        self.block2 = Block(512, 512)
        self.out = nn.Linear(512, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.out(x)
        return x
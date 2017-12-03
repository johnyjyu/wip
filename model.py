"""Implementation of the VQVAE network"""

import torch
import torch.nn as nn
from torch.autograd import Variable

# VectorQuantization from: https://github.com/JACKHAHA363/VQVAE/blob/master/model.py
class VectorQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, emb):
        """
        x: (bz, D)
        emb: (emb_num, D)
        output: (bz, D)
        """
        dist = row_wise_distance(x, emb)
        indices = torch.min(dist, -1)[1]
        ctx.indices = indices
        ctx.emb_num = emb.size(0)
        ctx.bz = x.size(0)
        return torch.index_select(emb, 0, indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.indices.view(-1,1)
        bz = ctx.bz
        emb_num = ctx.emb_num
        # get an one hot index
        one_hot_ind = torch.zeros(bz, emb_num)
        one_hot_ind.scatter_(1, indices, 1)
        one_hot_ind = Variable(one_hot_ind, requires_grad=False)
        grad_emb = torch.mm(one_hot_ind.t(), grad_output)
        return grad_output, grad_emb

def row_wise_distance(v1, v2):
    """
    v1: (a, p)
    v2: (b, p)
    return: dist: (a, b), where dist[i,j] = l2_dist(m1[i], m2[j])
    """
    a = v1.size(0)
    b = v2.size(0)
    v1 = torch.stack([v1]*b).transpose(0,1)
    v2 = torch.stack([v2]*a)
    return torch.sum((v1-v2)**2, 2).squeeze()


class VQVAE(nn.Module):
    def __init__(self, input_dim, embed_dim, embed_num):
        super(VQVAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, embed_dim)
        self.fc3 = nn.Linear(embed_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.embed = nn.Embedding(embed_num, embed_dim)

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        z_e = self.fc2(h1)
        return z_e 

    def vq(self, z_e):
        z_q = VectorQuantization.apply(z_e, self.embed.weight) 
        return z_q

    def decode(self, z_q):
        h3 = self.relu(self.fc3(z_q))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        self.z_e = self.encode(x)
        self.z_q = self.vq(self.z_e)
        self.x_reconst = self.decode(self.z_q)
        return self.x_reconst

    def get_embed_weight(self):
        return self.embed.weight


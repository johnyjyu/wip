"""Implementation of the VQVAE network"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# VectorQuantization from: https://github.com/JACKHAHA363/VQVAE/blob/master/model.py
class VectorQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, emb):
        """
        x: (bz, D)
        emb: (emb_num, D)
        output: (bz, D)
        """
        """
        v1: (a, p)
        v2: (b, p)
        dist: (a, b), where dist[i,j] = l2_dist(m1[i], m2[j])
        """
        a = x.size(0)
        b = emb.size(0)
        v1 = torch.stack([x]*b).transpose(0,1)
        v2 = torch.stack([emb]*a)
        dist = torch.sum((v1-v2)**2, 2).squeeze()
        indices = torch.min(dist, -1)[1]
        ctx.indices = indices
        ctx.emb_num = emb.size(0)
        ctx.bz = x.size(0)

        #self.save_for_backward(x)
        result = torch.index_select(emb, 0, indices)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.indices.view(-1,1)
        bz = ctx.bz
        emb_num = ctx.emb_num
        # get an one hot index
        one_hot_ind = torch.zeros(bz, emb_num)
        if torch.cuda.is_available:
            one_hot_ind = one_hot_ind.cuda()
        one_hot_ind.scatter_(1, indices, 1)
        one_hot_ind = Variable(one_hot_ind, requires_grad=False)

        grad_input = grad_output.clone()
        grad_emb = torch.mm(one_hot_ind.t(), grad_output)
        '''
        zeros_hot_ind = Variable(torch.zeros(bz, emb_num), requires_grad=False)
        if torch.cuda.is_available:
            zeros_hot_ind = zeros_hot_ind.cuda()
        grad_emb_zeros = torch.mm(zeros_hot_ind, grad_output)
        '''
        result, = ctx.saved_variables
        return grad_input, grad_emb


class VQVAE(nn.Module):
    def __init__(self, input_dim, embed_dim, embed_num, batch_size):
        super(VQVAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, embed_dim)
        self.fc3 = nn.Linear(embed_dim, 400)
        self.fc4 = nn.Linear(400, 784)

        self.vqlayer = VectorQuantization()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.embed = nn.Embedding(embed_num, embed_dim)

        self.batch_size = batch_size

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        z_e = self.fc2(h1)
        return z_e 

    def vq(self, z_e):
        z_q = self.vqlayer.apply(z_e, self.embed.weight) 
        return z_q

    def decode(self, z_q):
        h3 = self.relu(self.fc3(z_q))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        self.z_e = self.encode(x.view(-1, 784))
        self.z_q = self.vq(self.z_e)
        self.x_reconst = self.decode(self.z_q)

        # reconstruction loss
        reconst_loss = F.binary_cross_entropy(self.x_reconst, x)
        # embedding loss
        detach_z_e = Variable(self.z_e.data, requires_grad=False)
        #z_q = model.vq(detach_z_e)
        z_q = self.z_q
        embed_loss= torch.sum((detach_z_e - z_q).pow(2))
        embed_loss /= self.batch_size
        # commitment loss
        detach_z_q = Variable(self.z_q.data, requires_grad=False)
        commit_loss = torch.sum((self.z_e - detach_z_q).pow(2))
        commit_loss /= self.batch_size

        return self.x_reconst, reconst_loss, embed_loss, commit_loss

    def get_embed_weight(self):
        return self.embed.weight

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

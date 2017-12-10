"""Implementation of the VQVAE network"""

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# VectorQuantization from: https://github.com/JACKHAHA363/VQVAE/blob/master/model.py
class VectorQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_e, emb):
        """
        z_e: (bz, D) i.e. 128 * 500
        emb: (emb_num, D) i.e. 10 * 500
        output: (bz, D) i.e. 128 * 500
        """
        """
        v1: (a, p)
        v2: (b, p)
        dist: (a, b), where dist[i,j] = l2_dist(m1[i], m2[j])
        """
        a = z_e.size(0) # 128
        b = emb.size(0) # 10
        v1 = torch.stack([z_e]*b).transpose(0,1) # 128*10*500
        v2 = torch.stack([emb]*a) # 128*10*500
        dist = torch.sum((v1-v2)**2, 2).squeeze() #128*10
        indices = torch.min(dist, -1)[1] # 128 
        ctx.indices = indices # 128 
        ctx.emb_num = emb.size(0) # 10
        ctx.bz = z_e.size(0) # 128
        #self.save_for_backward(z_e)
        z_q = torch.index_select(emb, 0, indices) # 128 * 500
        ctx.save_for_backward(z_q)
        return z_q

    @staticmethod
    def backward(ctx, grad_output):
        """grad_output: (128*500)  return: (128*500), (10*500)"""
        indices = ctx.indices.view(-1,1) # 128 * 1
        bz = ctx.bz # 128
        emb_num = ctx.emb_num # 10
        # get an one hot index
        one_hot_ind = torch.zeros(bz, emb_num) # 128 * 10
        if torch.cuda.is_available:
            one_hot_ind = one_hot_ind.cuda()
        one_hot_ind.scatter_(1, indices, 1) # 128 * 10
        one_hot_ind = Variable(one_hot_ind, requires_grad=False)
        grad_input = grad_output.clone() # 128 * 500
        grad_emb = torch.mm(torch.transpose(one_hot_ind,0,1), grad_output) #10*500
        #grad_emb = torch.mm(one_hot_ind.t(), grad_output) #10*500
        result, = ctx.saved_variables
        #return grad_input, None
        return None, grad_emb
        #return grad_input, grad_emb


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
        self.z_q = None

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        z_e = self.fc2(h1)
        return z_e 

    def vq(self, z_e, embed):
        weight = embed.weight
        z_q = self.vqlayer.apply(z_e, weight) 
        return z_q

    def decode(self, z_q):
        h3 = self.relu(self.fc3(z_q))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        self.z_e = self.encode(x.view(-1, 784))
        org_z = self.z_e
        self.z_q = self.vq(self.z_e, self.embed)
        z_e = self.z_e
        z_q_org = self.z_q
        #print(z_q_org.data)
        #print(z_q_org)
        #sys.exit()
        z_q = z_q_org.detach()
        z_q = Variable(z_q.data, requires_grad=True)
        z_q = z_q.permute(0,1)
        def hook(grad):
            nonlocal org_z
            self.saved_grad = grad     ## copying gradients from decoder input to encoder output
            self.saved_h = org_z
            return grad
        z_q.register_hook(hook)
        self.x_reconst = self.decode(z_q)
        #self.x_reconst = self.decode(self.z_q)

        # reconstruction loss
        reconst_loss = F.binary_cross_entropy(self.x_reconst, x)
        # embedding loss
        detach_z_e = Variable(self.z_e.data, requires_grad=False)
        #detach_z_e = self.z_e.detach()
        #embed_loss = torch.dist(detach_z_e, z_q)
        #embed_loss = embed_loss.sum(1).mean()
        embed_dist = (detach_z_e - z_q_org).pow(2)
        embed_loss= torch.sum(embed_dist)
        embed_loss /= self.batch_size
        # print(embed_loss.data)
        # commitment loss
        detach_z_q = Variable(self.z_q.data, requires_grad=False)
        #commit_loss = torch.dist(self.z_e, detach_z_q)
        #commit_loss = commit_loss.sum(1).mean()
        commit_loss = torch.sum((self.z_e - detach_z_q).pow(2))
        commit_loss /= self.batch_size

        return self.x_reconst, reconst_loss, embed_loss, commit_loss

    # back propagation for encoder part where we flow the saved gradients of the decoder through the encoder
    def bwd(self):
        self.saved_h.backward(self.saved_grad)

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

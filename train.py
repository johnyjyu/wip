"""Main pytorch-vqvae training script"""

from __future__ import print_function
import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import VQVAE


parser = argparse.ArgumentParser(description='VQVAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--input-dim', default=784, type=int)
parser.add_argument('--emb-dim', default=500, type=int)
parser.add_argument('--emb-num', default=10, type=int)
parser.add_argument('--beta', default=0.3, type=float)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


model = VQVAE(args.input_dim, args.emb_dim, args.emb_num)
if args.cuda:
    model.cuda()


def get_losses(recon_x, x, model):
    # reconstruction loss
    reconst_loss = F.binary_cross_entropy(recon_x, x)
    # embedding loss
    detach_z_e = Variable(model.z_e.data, requires_grad=False)
    z_q = model.vq(detach_z_e)
    embed_loss= torch.sum((detach_z_e - z_q).pow(2))
    embed_loss /= args.batch_size
    # commitment loss
    detach_z_q = Variable(model.z_q.data, requires_grad=False)
    commit_loss = torch.sum((model.z_e - detach_z_q).pow(2))
    commit_loss /= args.batch_size

    return reconst_loss, embed_loss, commit_loss


optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    """run one epoch of model to train with data loader"""
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data).view(-1, 784)
        if args.cuda:
            data = data.cuda()
        # run forward
        recon_batch = model(data)

        # compute losses
        reconst_loss, embed_loss, commit_loss = get_losses(recon_batch, data, model)
        # clear gradients and run backward
        optimizer.zero_grad()
        # get gradients for decoder and encoder
        loss = reconst_loss + args.beta * commit_loss
        loss.backward()
        # clear gradients in VQ embedding 
        model.embed.zero_grad()
        # get gradients for embedding
        embed_loss.backward()
        loss += embed_loss

        # run optimizer to update parameters
        optimizer.step()
        train_loss += loss.data[0]

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True).view(-1, 784)
        recon_batch = model(data)
        reconst_loss, embed_loss, commit_loss = get_losses(recon_batch, data, model)
        test_loss += (reconst_loss + embed_loss + args.beta*commit_loss).data[0]
        '''
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    '''
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


os.makedirs('results', exist_ok=True)
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

    # samples from discrete vectors
    embed_weight = model.get_embed_weight()
    if args.cuda:
       sample = sample.cuda()
    sample = model.decode(embed_weight).cpu()
    save_image(sample.data.view(args.emb_num, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')


from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import DoANet
from dataset import Dataset, collate_fn
from torch.autograd import Variable
from pprint import pprint
import time, os
from os.path import join, splitext, basename, dirname

# Training settings
parser = argparse.ArgumentParser(description='Speaker Embedding Network')
parser.add_argument('--kindex', type=int, default=1, metavar='N',
                    help='index for k-fold cross validation')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-p', '--postfix', type=str,default='./',
                    help='Path where results will be saved')
args = parser.parse_args()

# Preset
args.cuda = torch.cuda.is_available()

torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)


kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

assert args.postfix is not None, "You have to set postfix"

today = time.strftime('%y%m%d')
savepath = join('result', '{}{}'.format(today, args.postfix))
if not os.path.exists(savepath):
    os.makedirs(savepath)
elif args.postfix=='test':
    os.system("rm -rf {}/*".format(savepath))
else:
    input("Path already exists, wish to continue?")
    os.system("rm -rf {}/*".format(savepath))


def main_loop(which_set):

    model.eval()

    celoss = nn.CrossEntropyLoss()

    epoch_loss = 0
    epoch_correct = 0

    for batch_idx, (data, gt, fpath) in enumerate(loader[which_set]):
        if args.cuda:
            data, gt = data.cuda(), gt.cuda()

        optimizer.zero_grad()
        pred = model(data)
        loss = celoss(pred, gt)

        epoch_loss += loss.data
        pred = torch.argmax(pred, dim=1)
        epoch_correct += (pred == gt).sum()

        with open(join(savepath, 'loss_{}.txt'.format(which_set)), 'a') as f:
            epoch_loss = epoch_loss / len(loader[which_set])
            f.write('{}\n'.format(epoch_loss))

        with open(join(savepath, 'acc_{}.txt'.format(which_set)), 'a') as f:
            epoch_acc = float(epoch_correct) / len(loader[which_set].dataset)
            f.write('{}\n'.format(epoch_acc))

        stdout = 'Test result: [{:5.0f}/{:5.0f} ({:2.0f}%)]\tLoss: {:5.3f} Acc: {:5.3f}'.format(epoch_correct,
                                                len(loader[which_set].dataset), 100*epoch_acc, epoch_loss, epoch_acc)

        print(stdout)

        with open(join(savepath, 'total.txt'), 'a') as f:
            print_tuple = [xx for xx in sorted(zip([xx[-14:] for xx in fpath], gt.cpu().numpy(), pred.cpu().numpy()))]
            print_gt = [int(xx) for xx in gt]
            print_pred = [int(xx) for xx in pred]
            f.write('{}\n'.format(stdout))
            f.write('{}\n'.format(print_gt))
            f.write('{}\n'.format(print_pred))
            f.write('{}\n\n'.format(print_tuple))


if __name__=='__main__':

    loader = {
        'valid': torch.utils.data.DataLoader(Dataset(which_set='valid', kfold=5, kindex=args.kindex),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)
    }

    model = DoANet()
    model_path = join('./result')
    model_path = model_path + '/least_valid_loss_model.pt'
    model.load_state_dict(torch.load(model_path))

    if args.cuda:
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    main_loop(which_set='valid')

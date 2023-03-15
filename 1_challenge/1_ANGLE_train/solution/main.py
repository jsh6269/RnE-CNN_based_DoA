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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

min_train_loss = 1e8
min_valid_loss = 1e8

# Training settings
parser = argparse.ArgumentParser(description='Speaker Embedding Network')
parser.add_argument('--kindex', type=int, default=1, metavar='N',
                    help='index for k-fold cross validation')
parser.add_argument('--batch-size', type=int, default=308, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=308, metavar='N',
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


kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}



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


def main_loop(epoch, which_set):

    global min_train_loss
    global min_valid_loss

    if which_set == 'train':
        model.train()
    else:
        model.eval()

    celoss = nn.CrossEntropyLoss()

    if which_set == 'valid':
        epoch_loss = 0
        epoch_correct = 0

    for batch_idx, (data, gt, fpath) in enumerate(loader[which_set]):
        if args.cuda:
            data, gt = data.cuda(), gt.cuda()

        optimizer.zero_grad()
        pred = model(data)
        loss = celoss(pred, gt)

        if which_set == 'train':
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), join(savepath, 'model.pt'))

        if batch_idx % args.log_interval == 0 and which_set == 'train':
            with open(join(savepath, 'loss_{}.txt'.format(which_set)), 'a') as f:
                f.write('{}\t{}\n'.format(epoch + float(batch_idx)/len(loader[which_set]), loss.data))

            pred = torch.argmax(pred, dim=1)
            corr = (pred == gt).sum().float()
            acc = corr / len(pred)
            with open(join(savepath, 'acc_{}.txt'.format(which_set)), 'a') as f:
                f.write('{}\t{}\n'.format(epoch + float(batch_idx)/len(loader[which_set]), acc))

            stdout = 'Train Epoch: {} [{:5.0f}/{:5.0f} ({:2.0f}%)]\tLoss: {:.3f} Acc: {:.3f}'.format(
                epoch+1, corr, len(pred),
                100. * acc, loss.data, acc)

            with open(join(savepath, 'total.txt'), 'a') as f:
                f.write('{}  {}  {}\n'.format(which_set, epoch+1, stdout))

            if min_train_loss > loss.data:
                torch.save(model.state_dict(), join(savepath, 'least_train_loss_model.pt'))
                min_train_loss = loss.data
                # print('t_best saved')
            
            print(stdout)

        if which_set == 'valid':
            epoch_loss += loss.data

            pred = torch.argmax(pred, dim=1)
            epoch_correct += (pred == gt).sum()

    if which_set == 'valid':
        with open(join(savepath, 'loss_{}.txt'.format(which_set)), 'a') as f:
            epoch_loss = epoch_loss / len(loader[which_set])
            f.write('{}\t{}\n'.format(epoch, epoch_loss))

        with open(join(savepath, 'acc_{}.txt'.format(which_set)), 'a') as f:
            epoch_acc = float(epoch_correct) / len(loader[which_set].dataset)
            f.write('{}\t{}\n'.format(epoch, epoch_acc))

        stdout = 'Valid Epoch: {} [{:5.0f}/{:5.0f} ({:2.0f}%)]\tLoss: {:5.3f} Acc: {:5.3f}'.format(epoch, epoch_correct,
                                                len(loader[which_set].dataset), 100*epoch_acc, epoch_loss, epoch_acc)

        print(stdout)

        with open(join(savepath, 'total.txt'), 'a') as f:
            print_tuple = [xx for xx in sorted(zip([xx[-14:] for xx in fpath], gt.cpu().numpy(), pred.cpu().numpy()))]
            print_gt = [int(xx) for xx in gt]
            print_pred = [int(xx) for xx in pred]
            f.write('{}  {}  {}\n'.format(which_set, epoch, stdout))
            f.write('{}\n'.format(print_gt))
            f.write('{}\n'.format(print_pred))
            f.write('{}\n\n'.format(print_tuple))

        if min_valid_loss > epoch_loss:
            min_valid_loss= epoch_loss
            torch.save(model.state_dict(), join(savepath, 'least_valid_loss_model.pt'))
            # print('v_best saved')


if __name__=='__main__':

    loader = {
        'train': torch.utils.data.DataLoader(Dataset(which_set='train'),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs),
        'valid': torch.utils.data.DataLoader(Dataset(which_set='valid'),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)
    }

    model = DoANet()

    if args.cuda:
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        main_loop(epoch, which_set='valid')
        main_loop(epoch, which_set='train')

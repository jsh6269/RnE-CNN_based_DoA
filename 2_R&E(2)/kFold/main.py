import os, random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch.backends import cudnn
from model import DoANet
from dataset import Dataset
from os.path import join

def main_loop(epoch, which_set, args, savepath):

    global min_train_loss
    global min_valid_loss

    if which_set == 'train':
        model.train()
    else:
        model.eval()

    celoss = nn.CrossEntropyLoss()
    epoch_loss = 0
    epoch_correct = 0
    ang_error = 0

    for batch_idx, (data, gt, fpath) in enumerate(loader[which_set]):
        if args.cuda:  data, gt = data.cuda(), gt.cuda()
        optimizer.zero_grad()
        pred = model(data)
        loss = celoss(pred, gt)

        if which_set == 'train':
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), join(savepath, 'model.pt'))

        if which_set == 'train':
            # saving train result in txt file
            num_tbatch = len(loader['train'])
            with open(join(savepath, 'loss_train.txt'), 'a') as f:
                f.write('{:.2f}\t{:.5f}\n'.format(epoch + float(batch_idx) / num_tbatch, loss.data))

            pred = torch.argmax(pred, dim=1)
            corr = (pred == gt).sum().float()
            acc = corr / len(pred)
            ang_error = abs(pred-gt).sum().float()/len(pred)
            ang_error *= 20

            with open(join(savepath, 'acc_train.txt'), 'a') as f:
                f.write('{:.2f}\t{:.5f}\n'.format(epoch + float(batch_idx) / num_tbatch, acc))

            stdout = 'Train Epoch: {} [{:5.0f}/{:5.0f} ({:2.0f}%)]\tLoss: {:.3f} Acc: {:.3f}'.format(
                    epoch + 1, corr, len(pred), 100. * acc, loss.data, acc)
            print(stdout)

            with open(join(savepath, 'train_ang_error.txt'), 'a') as f:
                f.write('{:.2f}\t{:.5f}\n'.format(epoch + float(batch_idx) / num_tbatch, ang_error))

            with open(join(savepath, 'total.txt'), 'a') as f:
                f.write('train  {}  {}\n'.format(epoch + 1, stdout))
                # f.write(str(pred))
                # f.write(str(gt))

            if min_train_loss > loss.data:
                torch.save(model.state_dict(), join(savepath, 'least_train_loss_model.pt'))
                min_train_loss = loss.data

        if which_set == 'valid':
            epoch_loss += loss.data
            pred = torch.argmax(pred, dim=1)
            epoch_correct += (pred == gt).sum()
            ang_error += abs(pred-gt).sum().float()


    if which_set == 'valid':
        # saving valid result in txt file
        with open(join(savepath, 'loss_valid.txt'), 'a') as f:
            epoch_loss = epoch_loss / len(loader['valid'])
            f.write('{}\t{:.5f}\n'.format(epoch, epoch_loss))

        with open(join(savepath, 'acc_valid.txt'), 'a') as f:
            epoch_acc = float(epoch_correct) / len(loader['valid'].dataset)
            f.write('{}\t{:.5f}\n'.format(epoch, epoch_acc))

        stdout = 'Valid Epoch: {} [{:5.0f}/{:5.0f} ({:2.0f}%)]\tLoss: {:5.3f} Acc: {:5.3f}'.format(
            epoch, epoch_correct, len(loader[which_set].dataset), 100*epoch_acc, epoch_loss, epoch_acc)
        print(stdout)

        ang_error = ang_error / len(loader['valid'].dataset)
        ang_error *= 20

        with open(join(savepath, 'valid_ang_error.txt'), 'a') as f:
            f.write('{:.2f}\t{:.5f}\n'.format(epoch, ang_error))

        with open(join(savepath, 'total.txt'), 'a') as f:
            f.write('{}  {}  {}\n'.format(which_set, epoch, stdout))

        if min_valid_loss > epoch_loss:
            min_valid_loss= epoch_loss
            torch.save(model.state_dict(), join(savepath, 'least_valid_loss_model.pt'))

def run_kfold(kindex, args, superpath):
    global model, loader, optimizer
    global min_train_loss
    global min_valid_loss
    global savepath

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    min_train_loss = 1e8
    min_valid_loss = 1e8

    loader = {
        'train': torch.utils.data.DataLoader(Dataset(which_set='train', kfold=args.kfold, kindex=kindex,
        datapath= args.datapath), batch_size=args.batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(Dataset(which_set='valid', kfold=args.kfold, kindex=kindex,
        datapath = args.datapath), batch_size=args.batch_size, shuffle=True),
    }

    model = DoANet(args.input_channel)
    if args.cuda: model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    savepath = superpath + 'index' + str(kindex)
    os.makedirs(savepath)

    for epoch in range(args.epochs):
        main_loop(epoch, 'valid', args, savepath)
        main_loop(epoch, 'train', args, savepath)

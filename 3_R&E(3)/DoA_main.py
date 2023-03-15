import time, os, random, math
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparsep
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from DoA_model import DoANet
from DoA_dataset import Dataset
from Tool_kit import *
from os.path import join
from glob import glob

parser = argparse.ArgumentParser(description='Direction Of Arrival Network')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--savepath', type=str, default='default')
parser.add_argument('--kindex', type=int, default=1)
parser.add_argument('--kfold', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--testPt', type=str, default='./previous_pt/bestmodel.pt')
parser.add_argument('--vadPred', type=str, default='./vadPred/vadPred3.pt')
parser.add_argument('--npPath', type=list, default=[
    './rneData2/8XAudioNoise/drone1'
])
parser.add_argument('--validate', type=list, default=[
    './rneData2/8XAudioNoise/drone1'
])
parser.add_argument('--fileNum', type=int, default=1160)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--pin_memory', type=bool, default=True)
parser.add_argument('--new_mode', type=str, default="slice")

args = parser.parse_args()

for folder in np.sort(glob('./rneData2/*XAudioNoise/*')):
    if '.npy' in folder:
        continue
    if 'street' not in folder and 'Siren' not in folder:
        args.npPath.append(folder)

# end

args.cuda = torch.cuda.is_available()
kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

min_loss = 1e8
start = time.time()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.mode == 'continue' and args.savepath == 'default':
    args.savepath = np.sort(glob('./result/*'))[-1]

elif args.mode == 'train':
    today = time.strftime('%y%m%d (%Hh %Mm %Ss)')
    args.savepath = join('result', '{}{}'.format(today, './'))
    os.makedirs(args.savepath)


def main_loop(epoch, which_set):

    global min_loss
    global start

    if which_set == 'train':
        model.train()
    else:
        model.eval()

    celoss = nn.CrossEntropyLoss()
    epoch_loss = 0
    epoch_correct = 0
    count = 0

    result_dict = {}
    # = {
    #   3 : {'gt' : 2, 'pred' : [?, ?, ?, ?]}
    # }
    #
    # if id=3, gt=2, pred=[?, ?, ?, ?]

    for batch_idx, (data, gt, id) in enumerate(loader[which_set]):
        # data ; [batch, 2, 80, 6]   // batch = 232 (example)
        # gt ; [batch]

        if args.cuda:
            data, gt = data.cuda(), gt.cuda()

        optimizer.zero_grad()
        pred = model(data)
        loss = celoss(pred, gt)
        epoch_loss += loss.data
        iter_correct = 0
        iter_loss = loss.data

        if which_set == 'train':
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), join(args.savepath, 'model.pt'))
        pred = torch.argmax(pred, dim=1)

        for i in range(len(pred)):
            if pred[i] == gt[i]:
                iter_correct += 1
                epoch_correct += 1
            if int(id[i]) not in result_dict.keys():
                result_dict[int(id[i])] = {}
                result_dict[int(id[i])]['gt'] = int(gt[i])
                result_dict[int(id[i])]['pred'] = []

            result_dict[int(id[i])]['pred'].append(int(pred[i]))

        # if which_set == 'train':
        #     print_out_doa('train_iteration', epoch, iter_correct, data.shape[0], iter_loss, time.time()-start)
        #     start = time.time()

    with torch.no_grad():
        epoch_loss = epoch_loss / len(loader[which_set])
        print_out_doa(which_set, epoch, epoch_correct, len(loader[which_set].dataset), epoch_loss, time.time()-start, args.savepath, dict=result_dict)
        start = time.time()

        if which_set == 'valid' and min_loss > epoch_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), join(args.savepath, 'min_loss_model.pt'))


if __name__=='__main__':

    if args.new_mode != 'origin':
        args.vadPred = torch.load(args.vadPred)

    loader = {
        'train': torch.utils.data.DataLoader(Dataset(which_set='train', kfold=args.kfold, kindex=args.kindex,
        npPath=args.npPath, gpu=args.cuda, vadPred=args.vadPred, mode=args.new_mode), batch_size=args.batch_size, shuffle=True, **kwargs),
        'valid': torch.utils.data.DataLoader(Dataset(which_set='valid', kfold=args.kfold, kindex=args.kindex,
        npPath=args.validate, gpu=args.cuda, vadPred=args.vadPred, mode=args.new_mode), batch_size=args.batch_size, shuffle=True, **kwargs)
    }

    model = DoANet()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    info_writing(__file__, DoANet, args.savepath)

    if args.mode == 'continue':
        model.load_state_dict(torch.load(join(args.savepath, 'model.pt')))

    if args.mode == 'train' or args.mode == 'continue':
        start_epoch = get_next_epoch(args.savepath)
        for epoch in range(start_epoch, args.epochs):
            with torch.no_grad():
                main_loop(epoch, which_set='valid')
            main_loop(epoch, which_set='train')
            if epoch != 0 and epoch % 5 == 0:
                result_plot(args.savepath)

    if args.mode == 'test':
        if args.cuda:
            model.load_state_dict(torch.load(args.testPt))
        else:
            model.load_state_dict(torch.load(args.testPt, map_location='cpu'))
        main_loop(1, which_set='test')

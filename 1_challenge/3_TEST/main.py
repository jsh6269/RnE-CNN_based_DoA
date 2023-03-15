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
from model1 import DoANet1
from model2 import DoANet2
from dataset import Dataset, collate_fn
from torch.autograd import Variable
from pprint import pprint
import time, os, json
from os.path import join, splitext, basename, dirname
from operator import itemgetter
data = {}
noVoice_list = []
angle = []
final_data = {}
for i in range (10):
    angle.append([])

# Training settings
parser = argparse.ArgumentParser(description='Speaker Embedding Network')
parser.add_argument('--kindex', type=int, default=1, metavar='N',
                    help='index for k-fold cross validation')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 1000)')
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

    for batch_idx, (data, fpath) in enumerate(loader["valid"]):
        if args.cuda:
            data = data.cuda()

        pred = model(data)

        pred = torch.argmax(pred, dim=1)

        if which_set == "VAD":
            for i in range((len(pred))):
                if pred[i]==0:
                    noVoice_list.append(i)

        if which_set == "ANGLE":
            for i in range((len(pred))):
                if(i not in noVoice_list):
                    angle[int(pred[i])].append(i)

        with open(join(savepath, 'total.txt'), 'a') as f:
            print_pred = [int(xx) for xx in pred]
            f.write('{}\n'.format(print_pred))

def write_VAD():
    global data
    data["track3_results"] = []
    for i in noVoice_list:
        temp = {"id": i, "angle": -1}
        data["track3_results"].append(temp)

def write_ANGLE():
    global data
    for i in range(10):
        for j in range(len(angle[i])):
            temp = {"id": angle[i][j], "angle": i*20}
            data["track3_results"].append(temp)

if __name__=='__main__':

    k = len(Dataset(which_set='valid'))
    loader = {
        'valid': torch.utils.data.DataLoader(Dataset(which_set='valid'),
        batch_size=k, shuffle=False, collate_fn=collate_fn, **kwargs)
    }
    
    model = DoANet1()
    model_path = './model/VAD_model.pt'
    model.load_state_dict(torch.load(model_path,map_location='cpu'))

    if args.cuda:
        model.cuda()

    main_loop(which_set='VAD')
    print(noVoice_list)
    write_VAD()

    model = DoANet2()
    model_path = './model/ANGLE_model.pt'
    if(args.cuda == False):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(model_path))

    if args.cuda:
        model.cuda()

    main_loop(which_set='ANGLE')
    write_ANGLE()
    a = sorted(data["track3_results"], key=lambda x : x["id"])
    #a = sorted(data["track3_results"], key=itemgetter("id"))
    final_data["track3_results"] = a
    with open('t3_res_[대전과학고].json', 'w') as outfile:
        json.dump(final_data, outfile, indent=4, sort_keys=True)


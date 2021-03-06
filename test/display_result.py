from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import sys
sys.path.append('./aux/')
from dataset import *
from model import *
from utils import *
from ply import *
import torch.nn.functional as F
import sys
from tqdm import tqdm
import os
import json
import time, datetime
import visdom
sys.path.append("./nndistance/")
from modules.nnd import NNDModule
distChamfer =  NNDModule()

# Parsing arguments
#------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 20000,  help='number of pts')
parser.add_argument('--nb_primitives', type=int, default = 10,  help='number of primitives in the atlas')
parser.add_argument('--super_pts', type=int, default = 2500,  help='number of input pts to pointNet, not used by default')
parser.add_argument('--env', type=str, default ="result",  help='visdom environment')
opt = parser.parse_args()
#-------------------------------------------------------------------------------


# Launch visdom for visualization
#-------------------------------------------------------------------------------
vis = visdom.Visdom(port = 8097, env=opt.env)
#-------------------------------------------------------------------------------

# Setting initial parameter
#-------------------------------------------------------------------------------
now                = datetime.datetime.now()
save_path          = opt.env
dir_name           = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname            = os.path.join(dir_name, 'log.txt')
blue               = lambda x:'\033[94m' + x + '\033[0m'
opt.manualSeed     = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
#-------------------------------------------------------------------------------


# Creating train/test dataloader
#-------------------------------------------------------------------------------
dataset      = ShapeNet( normal = False, class_choice = "chair", train=True)
dataset_test = ShapeNet( normal = False, class_choice = "chair", train=False)

dataloader      = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                              shuffle=None,
                                              num_workers=int(opt.workers))
dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=opt.batchSize,
                                              shuffle=False,
                                              num_workers=int(opt.workers))
cudnn.benchmark    = True
len_dataset        = len(dataset)
lrate              = 1.0
best_val_loss      = 10
lr_decay           = 70
constant_rep       = True
D3_is_on           = True
display_train      = 3
display_val        = 3
prim               = True
model              = "log/existance_prob_delaystart_3D/network.pth"
#-------------------------------------------------------------------------------

print('---------- Training information -----------')
print('Environment  : ', opt.env)
print('Training Set : ', len(dataset.datapath), 'elements')
print('Testing Set  : ', len(dataset_test.datapath), 'elements')
print("Random Seed  : ", opt.manualSeed)
print("Epoch        : ", opt.nepoch)
print("Primitives   : ", opt.nb_primitives)
print("points       : ", opt.num_points)
print("Batch size   : ", opt.batchSize)
print("3D on        : ", D3_is_on)
print('-------------------------------------------\n')

# Creating network
#-------------------------------------------------------------------------------
network = AE_AtlasNet(num_points    = opt.num_points,
                      nb_primitives = opt.nb_primitives,
                      D3_is_on      = D3_is_on)
network.cuda()
network.apply(weights_init)
if model != '':
    network.load_state_dict(torch.load(model))
    print(" Previous weight loaded ")
optimizer = optim.SGD(network.parameters(), lr = lrate)
#-------------------------------------------------------------------------------


# meters to record stats on learning
#-------------------------------------------------------------------------------
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f: #open and append
        f.write(str(network) + '\n')
#-------------------------------------------------------------------------------

with torch.no_grad():

    train_loss.reset()
    network.train()

    for i, data in enumerate(dataloader, 0):

        if i == display_train:
            break
        print(i,"display train")
        #read data
        #-----------------------------------------------------------------------
        _, pts, _, _, _  = data
        pts = Variable(pts)
        pts = pts.transpose(2,1).contiguous().cuda()
        #-----------------------------------------------------------------------

        #compute prediction
        #-----------------------------------------------------------------------
        config, config_prob, points_per_primitive = network(x=pts,
                                                 constant_rep=constant_rep)
        #-----------------------------------------------------------------------

        #display result
        #-----------------------------------------------------------------------
        label = 1
        labels_generated_pts = torch.Tensor()

        points_repartition = points_per_primitive[0]

        for j in range(points_repartition.size(0)):
            if(points_repartition[j] != 0):
                ones  = torch.ones(points_repartition[j].data[0])*label
                label = label + 1
                labels_generated_pts = torch.cat((labels_generated_pts,ones),0)

        vis.scatter(X = pts.transpose(2,1).contiguous()[0].data.cpu(),
                    win = 'train set input '+str(i),
                    opts = dict(title = "train set input "+str(i), markersize = 2,))

        x = []
        for j in range(points_per_primitive.size(1)):
            n = torch.ones(points_repartition[j].data[0])*j
            x.append(n)
        x = torch.cat(x,0)


        vis.histogram(X = x,
                      win = "repartition train "+str(i) ,
                      opts = dict(title = "repartition train", markersize = 2,),)

        vis.scatter(X = config[0].data.cpu(),
                    Y = labels_generated_pts,
                    win = 'train set ouput '+str(i),
                    opts = dict(title="train set output",markersize=2,),)

        if prim:
            X = config[0].data.cpu()
            current = 0
            for j in range(points_repartition.size(0)):
                points = points_repartition[j].data[0]
                if(points !=0):
                    title = "train primitive "+str(i)+"|"+str(j+1)+"("+str(points)+")"
                    xX  = X[current:current+points]
                    print(current,current+points)
                    opt = dict(title=title, markersize=2,)
                    vis.scatter(X = xX, win = title,opts = opt)
                    current=current+points
        #-----------------------------------------------------------------------

    network.eval()

    for i, data in enumerate(dataloader_test, 0):

        if i == display_val:
            break

        print(i,"display val")

        #read data
        #-----------------------------------------------------------------------
        _, pts, cat, _, _  = data
        pts = Variable(pts)
        pts = pts.transpose(2,1).contiguous().cuda()
        #-----------------------------------------------------------------------

        #compute prediction
        #-----------------------------------------------------------------------
        config, config_prob, points_per_primitive = network(x=pts,
                                                 constant_rep=constant_rep)
        #-----------------------------------------------------------------------

        #display result
        #-----------------------------------------------------------------------
        label = 1
        labels_generated_pts = torch.Tensor()

        points_repartition = points_per_primitive[0]

        for j in range(points_repartition.size(0)):
            if(points_repartition[j] != 0):
                ones  = torch.ones(points_repartition[j].data[0])*label
                label = label + 1
                labels_generated_pts = torch.cat((labels_generated_pts,ones),0)

        vis.scatter(X = pts.transpose(2,1).contiguous()[0].data.cpu(),
                    win = 'val set input '+str(i),
                    opts = dict(title = "val set input "+str(i), markersize = 2,),)

        x = []
        for j in range(points_per_primitive.size(1)):
            n = torch.ones(points_repartition[j].data[0])*j
            x.append(n)
        x = torch.cat(x,0)


        vis.histogram(X = x,
                      win = "repartition val "+str(i) ,
                      opts = dict(title = "repartition val", markersize = 2,),)

        vis.scatter(X = config[0].data.cpu(),
                    Y = labels_generated_pts,
                    win = 'val set ouput '+str(i),
                    opts = dict(title="val set output",markersize=2,),)

        if prim :
            X = config[0].data.cpu()
            current = 0
            for j in range(points_repartition.size(0)):
                points = points_repartition[j].data[0]
                if(points !=0):
                    title = "train primitive "+str(i)+"|"+str(j+1)+"("+str(points)+")"
                    xX  = X[current:current+points]
                    print(current,current+points)
                    opt = dict(title=title, markersize=2,)
                    vis.scatter(X = xX, win = title,opts = opt)
                    current=current+points
        #-----------------------------------------------------------------------

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
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points')
parser.add_argument('--nb_primitives', type=int, default = 25,  help='number of primitives in the atlas')
parser.add_argument('--super_points', type=int, default = 2500,  help='number of input points to pointNet, not used by default')
parser.add_argument('--env', type=str, default ="results_test",  help='visdom environment')
opt = parser.parse_args()
#-------------------------------------------------------------------------------


# Launch visdom for visualization
#-------------------------------------------------------------------------------
vis = visdom.Visdom(port = 8097, env=opt.env)
#-------------------------------------------------------------------------------

# Setting initial parameter
#-------------------------------------------------------------------------------
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')
blue = lambda x:'\033[94m' + x + '\033[0m'
opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss  = 10
display = 3
prim_display = 0
#-------------------------------------------------------------------------------


# Creating train/test dataloader
#-------------------------------------------------------------------------------
dataset      = ShapeNet( normal = False, class_choice = "chair", train=True)
dataset_test = ShapeNet( normal = False, class_choice = "chair", train=False)

dataloader      = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=None, num_workers=int(opt.workers))
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))
#-------------------------------------------------------------------------------

print('---------- Training information -----------')
print('Environment  : ', opt.env)
print('Training Set : ', len(dataset.datapath), 'elements')
print('Testing Set  : ', len(dataset_test.datapath), 'elements')
print("Random Seed  : ", opt.manualSeed)
print("Epoch        : ", opt.nepoch)
print("Primitives   : ", opt.nb_primitives)
print("Points       : ", opt.num_points)
print("Batch size   : ", opt.batchSize)
print('-------------------------------------------\n')

# Network setting
#-------------------------------------------------------------------------------
cudnn.benchmark = True
len_dataset     = len(dataset)
lrate = 0.001
#-------------------------------------------------------------------------------


# Creating network
#-------------------------------------------------------------------------------
network = AE_AtlasNet(num_points = opt.num_points, nb_primitives = opt.nb_primitives, outsize = 12)
network.cuda()
network.apply(weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
optimizer = optim.Adam(network.parameters(), lr = lrate)
#-------------------------------------------------------------------------------



# visdom
#-------------------------------------------------------------------------------
labels_generated_points = torch.Tensor(range(1, (opt.nb_primitives+1)*(opt.num_points/opt.nb_primitives)+1)).view(opt.num_points/opt.nb_primitives,(opt.nb_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.nb_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)
#-------------------------------------------------------------------------------


# Learning loop
#-------------------------------------------------------------------------------
print('---------------- Training -----------------')
it = 0
for i, data in enumerate(dataloader, 0):
    if i > 0:
            optimizer.zero_grad()
            img, points, cat, _, _ = data
            points                 = Variable(points)
            points                 = points.transpose(2,1).contiguous()
            points                 = points.cuda()
            pointsReconstructed    = network(points)

            # Visdom
            #-----------------------------------------------------------------------
            vis.scatter(X = points.transpose(2,1).contiguous()[0].data.cpu(),
                        win = 'training_inut_'+str(it),
                        opts = dict(title = 'training_input_'+str(it), markersize = 2,),)

            vis.scatter(X = pointsReconstructed[0].data.cpu(),
                        Y = labels_generated_points[0:pointsReconstructed.size(1)],
                        win = 'training_output_'+str(it),
                        opts = dict(title="training_output_"+str(it),markersize=2,),)

            print("Training - ",i)
            if it >= display :
                X = pointsReconstructed[0].data.cpu()
                k = opt.num_points/opt.nb_primitives
                c = np.array([[60,240,45]])

                for i in range(prim_display):
                        title = "training_output_5 Primitive - "+str(i)
                        xX    = X[i*k:(i+1)*k]
                        vis.scatter(xX,
                             win = title,
                             opts = dict(title=title,markersize=2,markercolor=c))
                break
            it = it + 1
            #-----------------------------------------------------------------------

# Learning loop
#-------------------------------------------------------------------------------
print('\n-------------- Validation -----------------')
it = 0
for i, data in enumerate(dataloader_test, 0):
    if i > 0:
            optimizer.zero_grad()
            img, points, cat, _, _ = data
            points                 = Variable(points)
            points                 = points.transpose(2,1).contiguous()
            points                 = points.cuda()
            pointsReconstructed    = network(points)

            # Visdom
            #-----------------------------------------------------------------------
            vis.scatter(X = points.transpose(2,1).contiguous()[0].data.cpu(),
                        win = 'validation_inut_'+str(it),
                        opts = dict(title = 'validation_input_'+str(it), markersize = 2,),)

            vis.scatter(X = pointsReconstructed[0].data.cpu(),
                        Y = labels_generated_points[0:pointsReconstructed.size(1)],
                        win = 'validation_output_'+str(it),
                        opts = dict(title="validation_output_"+str(it),markersize=2,),)

            print("Validation - ",i)
            if it >= display :
                X = pointsReconstructed[0].data.cpu()
                k = opt.num_points/opt.nb_primitives
                c = np.array([[60,240,45]])

                for i in range(prim_display):
                        title = "validation_output_5 Primitive - "+str(i)
                        xX    = X[i*k:(i+1)*k]
                        vis.scatter(xX,
                             win = title,
                             opts = dict(title=title,markersize=2,markercolor=c))
                break
            it = it + 1
            #-----------------------------------------------------------------------

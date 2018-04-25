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
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 10000,  help='number of points')
parser.add_argument('--nb_primitives', type=int, default = 25,  help='number of primitives in the atlas')
parser.add_argument('--super_points', type=int, default = 2500,  help='number of input points to pointNet, not used by default')
parser.add_argument('--env', type=str, default ="AE_AtlasNet_MatBias_2D_50E_25P_1024VL_LRdecay",  help='visdom environment')
opt = parser.parse_args()
#-------------------------------------------------------------------------------


# Launch visdom for visualization
#-------------------------------------------------------------------------------
vis = visdom.Visdom(port = 8097, env=opt.env)
#-------------------------------------------------------------------------------

# Setting initial parameter
#-------------------------------------------------------------------------------
now = datetime.datetime.now()
save_path = opt.env
dir_name = os.path.join('log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')
blue = lambda x:'\033[94m' + x + '\033[0m'
opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss  = 10
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
lrate           = 0.001
lr_decay        = 1
#-------------------------------------------------------------------------------


# Creating network
#-------------------------------------------------------------------------------
network = AE_AtlasNet(num_points = opt.num_points, nb_primitives = opt.nb_primitives)
network.cuda()
network.apply(weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
optimizer = optim.Adam(network.parameters(), lr = lrate)
#-------------------------------------------------------------------------------


# meters to record stats on learning
#-------------------------------------------------------------------------------
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f: #open and append
        f.write(str(network) + '\n')
#-------------------------------------------------------------------------------


# visdom
#-------------------------------------------------------------------------------
win_curve = vis.line(X = np.array([0]),Y = np.array([0]),)
val_curve = vis.line(X = np.array([0]),Y = np.array([1]),)
labels_generated_points = torch.Tensor(range(1, (opt.nb_primitives+1)*(opt.num_points/opt.nb_primitives)+1)).view(opt.num_points/opt.nb_primitives,(opt.nb_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.nb_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)
#-------------------------------------------------------------------------------



# Learning loop
#-------------------------------------------------------------------------------
for epoch in range(opt.nepoch):

    train_loss.reset()
    network.train()

    if(epoch == lr_decay):
        lr = lrate/10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print('---------------- Training -----------------')
    print('[epoch:   it/maxit] - train loss\n')


    for i, data in enumerate(dataloader, 0):


        optimizer.zero_grad()
        img, points, cat, _, _ = data
        points                 = Variable(points)
        points                 = points.transpose(2,1).contiguous()
        points                 = points.cuda()
        pointsReconstructed    = network(points)
        dist1, dist2           = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed) #loss function
        loss_net               = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_net.backward()
        train_loss.update(loss_net[0].data)
        optimizer.step()
        print('[%5d: %4d/%5d] - %f ' %(epoch, i, len_dataset/32, loss_net[0].data))

        # Visdom
        #-----------------------------------------------------------------------
        if (i==27):
            vis.scatter(X = points.transpose(2,1).contiguous()[0].data.cpu(),
                        win = 'TRAINING_SET_INPUT',
                        opts = dict(title = "TRAINING_SET_INPUT", markersize = 2,),)

            vis.scatter(X = pointsReconstructed[0].data.cpu(),
                        Y = labels_generated_points[0:pointsReconstructed.size(1)],
                        win = 'TRAINING_SET_OUTPUT',
                        opts = dict(title="TRAINING_SET_OUTPUT",markersize=2,),)

            X = pointsReconstructed[0].data.cpu()
            k = opt.num_points/opt.nb_primitives
            c = np.array([[60,240,45]])

            for i in range(5):
                    title = "Training Primitive - "+str(i)
                    xX    = X[i*k:(i+1)*k]
                    vis.scatter(xX,
                         win = title,
                         opts = dict(title=title,markersize=2,markercolor=c))

    # exit(0)
    if train_loss.avg != 0:
        vis.updateTrace(
            X = np.array([epoch]),
            Y = np.log(np.array([train_loss.avg])),
            win = win_curve,
            name = 'Chamfer train')
        #-----------------------------------------------------------------------




    # Validation
    #---------------------------------------------------------------------------
    print('\n-------------- Validation -----------------')
    print('[epoch:   it:maxit] - val loss\n')

    val_loss.reset()
    for item in dataset_test.cat:
        dataset_test.perCatValueMeter[item].reset()

    with torch.no_grad():
        network.eval()
        for i, data in enumerate(dataloader_test, 0):
            img, points, cat, _ , _ = data
            points               = Variable(points)
            points               = points.transpose(2,1).contiguous()
            points               = points.cuda()
            pointsReconstructed  = network(points)
            dist1, dist2         = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed)
            loss_net             = (torch.mean(dist1)) + (torch.mean(dist2))
            val_loss.update(loss_net[0].data)
            dataset_test.perCatValueMeter[cat[0]].update(loss_net[0].data)

            print('[%5d: %4d/%5d] - %f ' %(epoch, i, len(dataset_test), loss_net.data[0]))

            if i==6:
                vis.scatter(X = points.transpose(2,1).contiguous()[0].data.cpu(),
                        win = 'VALIDATION_SET_INPUT',
                        opts = dict(title = "VALIDATION_SET_INPUT", markersize = 2,),)
                vis.scatter(X = pointsReconstructed[0].data.cpu(),
                        Y = labels_generated_points[0:pointsReconstructed.size(1)],
                        win = 'VALIDATION_SET_OUTPUT',
                        opts = dict(title = "VALIDATION_SET_OUTPUT",markersize = 2,),)

                X = pointsReconstructed[0].data.cpu()
                k = opt.num_points/opt.nb_primitives
                c = np.array([[200,70,110]])

                for i in range(5):
                        title = "Validation Primitive - "+str(i)
                        xX    = X[i*k:(i+1)*k]
                        vis.scatter(xX,
                             win = title,
                             opts = dict(title=title,markersize=2,markercolor=c))
        if val_loss.avg != 0:
            vis.updateTrace(
                X = np.array([epoch]),
                Y = np.log(np.array([val_loss.avg])),
                win = val_curve,
                name = 'Chamfer val')
    #---------------------------------------------------------------------------

    # logging result
    #---------------------------------------------------------------------------
    log_table = {
      "train_loss"   : train_loss.avg,
      "val_loss"     : val_loss.avg,
      "epoch"        : epoch,
      "lr"           : lrate,
      "super_points" : opt.super_points,
      "bestval"      : best_val_loss,
    }

    print('\n------------------- Results ------------------\n')
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg[0])
   #     log_table.update({item: dataset_test.perCatValueMeter[item].avg[0]})
   # with open(logname, 'a') as f: #open and append
   #     f.write('json_stats: ' + json.dumps(log_table) + '\n')

    #save best network
    if best_val_loss > val_loss.avg:
        best_val_loss = val_loss.avg
        print('New best loss : ', best_val_loss)
        print('saving net...')
        torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
    #---------------------------------------------------------------------------

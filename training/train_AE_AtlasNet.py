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
import pickle
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
parser.add_argument('--num_points', type=int, default = 25000,  help='number of pts')
parser.add_argument('--nb_primitives', type=int, default = 125,  help='number of primitives in the atlas')
parser.add_argument('--super_pts', type=int, default = 2500,  help='number of input pts to pointNet, not used by default')
parser.add_argument('--env', type=str, default ="test",  help='visdom environment')
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
stop_constant_repartitionartition = 150
with3Dsurface                  = True
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
print("3D on        : ", with3Dsurface)
print('-------------------------------------------\n')

# Creating network
#-------------------------------------------------------------------------------
network = AE_AtlasNet(num_points    = opt.num_points,
                      nb_primitives = opt.nb_primitives,
                      with3Dsurface      = with3Dsurface)
network.cuda()
network.apply(weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
optimizer = optim.SGD(network.parameters(), lr = lrate)
#-------------------------------------------------------------------------------


# meters to record stats on learning
#-------------------------------------------------------------------------------
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
with open(logname, 'a') as f: #open and append
        f.write(str(network) + '\n')

log_train_loss = []
log_valid_loss = []
#-------------------------------------------------------------------------------


# visdom
#-------------------------------------------------------------------------------
win_curve = vis.line(X = np.array([0]),Y = np.array([0]),)
val_curve = vis.line(X = np.array([0]),Y = np.array([1]),)
#-------------------------------------------------------------------------------

constant_repartition = True

for epoch in range(opt.nepoch):

    train_loss.reset()
    network.train()

    #update the learning rate
    #---------------------------------------------------------------------------
    if((epoch%lr_decay == 0) and (epoch > 0)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2
    #---------------------------------------------------------------------------

    if stop_constant_repartitionartition == 0:
        constant_repartition = False

    if((epoch == stop_constant_repartitionartition) and (epoch > 0)):
        constant_repartition = False
z
    for i, data in enumerate(dataloader, 0):

        optimizer.zero_grad()

        #read data
        #-----------------------------------------------------------------------
        _, pts, _, _, _  = data
        pts = Variable(pts)
        pts = pts.transpose(2,1).contiguous().cuda()
        #-----------------------------------------------------------------------

        #compute prediction
        #-----------------------------------------------------------------------
        config, config_prob, points_per_primitive = network(x=pts,
                                                 constant_repartition=constant_repartition)


        print(config, points_per_primitive)
        exit(0)
        #-----------------------------------------------------------------------

        #compute the loss
        #-----------------------------------------------------------------------
        dist1, dist2 = distChamfer(pts.transpose(2,1).contiguous(),config)
        chamfer      = torch.mean(dist1) + torch.mean(dist2)
        dist1        = torch.mean(dist1,1) * config_prob
        dist2        = torch.mean(dist2,1) * config_prob
        loss_net     = torch.sum(dist1) + torch.sum(dist2)
        #-----------------------------------------------------------------------

        #display result
        #-----------------------------------------------------------------------
        print('[%5d: %5d/%5d] %f T'%(epoch,i,len_dataset/32,chamfer[0].data))
        if(i%50==0 and i !=0):
            display_result(pts,
                           config,
                           points_per_primitive,
                           epoch,
                           i,
                           vis,
                           "training")
        #-----------------------------------------------------------------------

        #backpropagate
        #-----------------------------------------------------------------------
        loss_net.backward()
        train_loss.update(chamfer[0].data)
        optimizer.step()
        #-----------------------------------------------------------------------

    #update the loss
    #---------------------------------------------------------------------------
    if train_loss.avg != 0:
        log_train_loss.append(train_loss.avg)
        vis.updateTrace(
            X = np.array([epoch]),
            Y = np.log(np.array([train_loss.avg])),
            win = win_curve,
            name = 'Chamfer train')
    #---------------------------------------------------------------------------

    val_loss.reset()

    for item in dataset_test.cat:
        dataset_test.perCatValueMeter[item].reset()

    with torch.no_grad():

        network.eval()

        for i, data in enumerate(dataloader_test, 0):

            #read data
            #-------------------------------------------------------------------
            _, pts, cat, _, _  = data
            pts = Variable(pts)
            pts = pts.transpose(2,1).contiguous().cuda()
            #-------------------------------------------------------------------

            #compute prediction
            #-------------------------------------------------------------------
            config, config_prob, points_per_primitive = network(x=pts,
                                                     constant_repartition=constant_repartition)
            #-------------------------------------------------------------------

            #compute the loss
            #-------------------------------------------------------------------
            dist1, dist2 = distChamfer(pts.transpose(2,1).contiguous(),config)
            chamfer      = torch.mean(dist1) + torch.mean(dist2)
            dist1        = torch.mean(dist1,1) * config_prob
            dist2        = torch.mean(dist2,1) * config_prob
            loss_net     = torch.sum(dist1) + torch.sum(dist2)
            #-------------------------------------------------------------------

            #-------------------------------------------------------------------
            val_loss.update(chamfer[0].data)
            dataset_test.perCatValueMeter[cat[0]].update(chamfer[0].data)
            #-------------------------------------------------------------------

            #display results
            #-------------------------------------------------------------------
            print('[%5d: %4d/%5d] %f V'%(epoch,i,len(dataset_test)/32,chamfer[0].data))
            if(i%5==0 and i !=0):
                display_result(pts,
                               config,
                               points_per_primitive,
                               epoch,
                               i,
                               vis,
                               "validation")
            #-------------------------------------------------------------------

    #update the loss
    #---------------------------------------------------------------------------
    if val_loss.avg != 0:
        log_valid_loss.append(val_loss.avg)
        vis.updateTrace(
            X = np.array([epoch]),
            Y = np.log(np.array([val_loss.avg])),
            win = val_curve,
            name = 'Chamfer val')
    #---------------------------------------------------------------------------

    print('\n----------------- Results -----------------')
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg[0])

    print('saving net...')
    torch.save(network.state_dict(), '%s/network.pth' % (dir_name))

    with open(dir_name+'/train-loss.pickle','wb') as f:
        pickle.dump(log_train_loss,f)

    with open(dir_name+'/val-loss.pickle','wb') as f:
        pickle.dump(log_valid_loss,f)
    #---------------------------------------------------------------------------

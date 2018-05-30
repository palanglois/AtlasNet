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
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 1000,  help='number of pts')
parser.add_argument('--nb_primitives', type=int, default = 10,  help='number of primitives in the atlas')
parser.add_argument('--super_pts', type=int, default = 2500,  help='number of input pts to pointNet, not used by default')
parser.add_argument('--env', type=str, default ="primitive_selection",  help='visdom environment')
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
                                              shuffle=False,
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
constant_rep       = False
D3_is_on           = True
display_train      = 5
display_val        = 4
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

def repartition(distribution):

    distrib    = distribution.clone()
    position   = 1
    classement = np.zeros(distrib.size(0),dtype=int)
    M          = torch.max(distrib,0)

    while(torch.sum(distrib) != 0):
        max_d, index = torch.max(distrib,0)

        if max_d[0] == M[0]:
            classement[index[0]] = position
        else:
            position = position + 1
            classement[index[0]] = position
            M = max_d

        distrib[index[0]] = 0

    classement[classement==0] = position + 1

    return np.array(classement)

def compare(distrib, target):

    common = distrib[distrib == target]
    non_zero = target[target != 0]
    return np.all(common==non_zero) and len(common) != 0 and len(non_zero) != 0

prim1  = 1
prim2  = 0
prim3  = 2
prim4  = 0
prim5  = 0
prim6  = 0
prim7  = 0
prim8  = 0
prim9  = 0
prim10 = 0

target_classement = np.array([prim1,prim2,prim3,prim4,prim5,
                              prim6,prim7,prim8,prim9,prim10])

found = 0
with torch.no_grad():

    train_loss.reset()
    network.train()

    for i, data in enumerate(dataloader, 0):

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

        for j in range(opt.batchSize):
            prim_classement = repartition(points_per_primitive[j])
            match = compare(prim_classement, target_classement)
            if(match):
                found = found + 1
                print("one matching configuration as been found (over "+str(i*2 + j)+")")

                #display result
                #-----------------------------------------------------------------------
                label = 1
                labels_generated_pts = torch.Tensor()

                points_repartition = points_per_primitive[j]

                for j in range(points_repartition.size(0)):
                    if(points_repartition[j] != 0):
                        ones  = torch.ones(points_repartition[j].data[0])*label
                        label = label + 1
                        labels_generated_pts = torch.cat((labels_generated_pts,ones),0)

                vis.scatter(X = pts.transpose(2,1).contiguous()[0].data.cpu(),
                            win = 'train set input '+str(2*i+j),
                            opts = dict(title = "train set input "+str(2*i+j), markersize = 2,))

                x = []
                for j in range(points_per_primitive.size(1)):
                    n = torch.ones(points_repartition[j].data[0])*j
                    x.append(n)
                x = torch.cat(x,0)


                vis.histogram(X = x,
                              win = "repartition train "+str(2*i+j) ,
                              opts = dict(title = "repartition train", markersize = 2,),)

                vis.scatter(X = config[0].data.cpu(),
                            Y = labels_generated_pts,
                            win = 'train set ouput '+str(2*i+j),
                            opts = dict(title="train set output",markersize=2,),)

                if(found == display_train):
                    print("limit of training config displaid has been reached, ggwp")
                    exit(0)
                # if prim:
                #     X = config[0].data.cpu()
                #     current = 0
                #     for j in range(points_repartition.size(0)):
                #         points = points_repartition[j].data[0]
                #         if(points !=0):
                #             title = "train primitive "+str(2*i+j)+"|"+str(j+1)+"("+str(points)+")"
                #             xX  = X[current:current+points]
                #             print(current,current+points)
                #             opt = dict(title=title, markersize=2,)
                #             vis.scatter(X = xX, win = title,opts = opt)
                #             current=current+points
                #-----------------------------------------------------------------------

            if(found == 0 and (i*2+j)%100 == 0 and (i*2+j)>0):
                print("none the training configuration match (over "+str(i*2+ j)+")")


    if found == 0:
        print("none of the training configuration match the target... sad")
    else:
        print(found,"configuration(s) has been found, ggwp")

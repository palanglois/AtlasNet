from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import adjust
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F

#UTILITIES
class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False, bottleneck_size = 1024,):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, bottleneck_size, 1)

        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size)
        self.trans = trans


        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, self.bottleneck_size)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, self.bottleneck_size, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

class PointNetfeatNormal(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeatNormal, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans


        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

#OUR METHOD
import resnet

class PointGenCon(nn.Module):
    #decode the output of the MLPs

    def __init__(self, bottleneck_size = 2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()

        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size/2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size/2, self.bottleneck_size/4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size/4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size/2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size/4)

    def forward(self, x):

        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))

        return x



class rotBias(nn.Module):
    def __init__(self, bottleneck_size = 2500, outsize = 12):

        super(rotBias, self).__init__()

        self.lin = torch.nn.Conv1d(bottleneck_size, outsize, 1)
        self.bn  = torch.nn.BatchNorm1d(outsize)
        self.th  = nn.Tanh()

    def forward(self, x):
        return self.th(self.bn(self.lin(x)))



class ExistanceProbability(nn.Module):
    def __init__(self, bottleneck_size = 2500, outsize = 40):

        super(ExistanceProbability, self).__init__()

        self.lin = torch.nn.Conv1d(bottleneck_size, outsize, 1)
        self.bn  = torch.nn.BatchNorm1d(outsize)

    def forward(self, x):
        return torch.nn.functional.softmax(self.bn(self.lin(x)),dim=1)



class d2Tod3(nn.Module):
    def __init__(self):

        super(d2Tod3, self).__init__()

        self.conv1 = torch.nn.Conv1d(2, 10, 1)
        self.conv3 = torch.nn.Conv1d(10, 3, 1)
        self.bn10  = torch.nn.BatchNorm1d(10)
        self.bn3   = torch.nn.BatchNorm1d(3)
        self.th    = nn.Tanh()

    def forward(self, x):
        if(x.size(2) > 1):
            x = F.relu(self.bn10(self.conv1(x)))
            x = self.th(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = self.th(self.conv3(x))
        return x


class AE_AtlasNet(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024,nb_primitives = 1,D3_is_on=True):
        super(AE_AtlasNet, self).__init__()

        self.num_points      = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives   = nb_primitives
        self.D3_is_on        = D3_is_on

        if self.D3_is_on:
            self.outsize = 12
        else:
            self.outsize = 9

        #encoder : 32(batch)x3(3D)x2500(points) -> 32(batch)x1024(latent vector)
        #-----------------------------------------------------------------------
        self.encoder = nn.Sequential(
        PointNetfeat(num_points,
                     global_feat=True,
                     trans = False,
                     bottleneck_size=self.bottleneck_size),)
        #-----------------------------------------------------------------------

        #define the different moduls used during the training
        #-----------------------------------------------------------------------
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = self.bottleneck_size) for i in range(0,self.nb_primitives)])
        self.existanceProbability = ExistanceProbability(bottleneck_size=self.bottleneck_size,outsize=self.nb_primitives)
        self.d2Tod3  = nn.ModuleList([d2Tod3() for i in range(0,self.nb_primitives)])
        self.rotBias = nn.ModuleList([rotBias(bottleneck_size = self.bottleneck_size, outsize = self.outsize) for i in range(0,self.nb_primitives)])
        #-----------------------------------------------------------------------

    def forward(self, x, std_val=0.01,constant_rep=True):

        #create the latent vector
        #-----------------------------------------------------------------------
        x = self.encoder(x)
        #-----------------------------------------------------------------------

        if constant_rep:
            N = self.num_points/self.nb_primitives
            ones = Variable(torch.ones(x.size(0),self.nb_primitives)).cuda()
            points_per_primitive  = ones.contiguous() * N
            points_per_primitive =points_per_primitive.int()
            existance_prob = ones[1]

        else :
            #compute the means for the normal distributions
            #-------------------------------------------------------------------
            mean = self.existanceProbability(x.unsqueeze(2))
            mean = mean.view(x.size(0),self.nb_primitives).contiguous()
            #-------------------------------------------------------------------

            #fixed standart deviation for every normal distriutions
            #-------------------------------------------------------------------
            ones    = Variable(torch.ones(x.size(0),self.nb_primitives)).cuda()
            stddev  = (std_val * ones).contiguous()
            #-------------------------------------------------------------------

            #sample over the normal distriutions the existance probabilities and
            #make sure they are in [0,1]
            #-------------------------------------------------------------------
            existance_prob = torch.normal(mean,stddev).contiguous()
            existance_prob[existance_prob < 0] = 0
            existance_prob[existance_prob > 1] = 1
            # existance_prob_ = existance_prob
            #-------------------------------------------------------------------

            # normalize the probabilities for every batch
            #-------------------------------------------------------------------
            batch_sum_prob = torch.sum(existance_prob,1).contiguous()
            batch_sum_prob = batch_sum_prob.expand(self.nb_primitives,x.size(0))
            batch_sum_prob = batch_sum_prob.permute(1,0)
            batch_sum_prob = batch_sum_prob.contiguous()
            existance_prob = torch.div(existance_prob,batch_sum_prob)
            existance_prob = existance_prob.contiguous()
            #-------------------------------------------------------------------

            #computing the number of points per primitives wrt the batch
            #-------------------------------------------------------------------
            points_per_primitive = torch.round(existance_prob*self.num_points)
            points_per_primitive = points_per_primitive.int().contiguous()
            points_per_primitive = adjust(points_per_primitive,self.num_points)
            #-------------------------------------------------------------------

        #create on spatial transformation per primitive
        #-----------------------------------------------------------------------
        linear_list = []
        for primitive in range(self.nb_primitives):
            linear_list.append(self.rotBias[primitive](x.unsqueeze(2).contiguous()))
        linear_list = torch.cat(linear_list,2)
        #-----------------------------------------------------------------------

        out_batch = []

        for batch in range(x.size(0)):

            out_primitive = []

            for primitive in range(self.nb_primitives):

                #find the number of points to generate wrt the batch and the
                #primitive number
                #---------------------------------------------------------------
                N = points_per_primitive[batch,primitive].data[0]
                #---------------------------------------------------------------

                #ignore the primitive is there is no point to associated
                #---------------------------------------------------------------
                if (N == 0):
                    continue
                #---------------------------------------------------------------

                #generete the 2D-primitive wrt the number of points
                #---------------------------------------------------------------
                rand_grid = Variable(torch.cuda.FloatTensor(1,2,N))
                rand_grid.data.uniform_(0,1)
                #---------------------------------------------------------------

                #allowing the network to modify the plan before the spatial
                #transformation
                #---------------------------------------------------------------
                if self.D3_is_on:

                    #tranform the 2D primitive into a 3D surface
                    #-----------------------------------------------------------
                    y = self.d2Tod3[primitive](rand_grid)
                    #-----------------------------------------------------------

                    #generate the roation matrix & bias wrt the primitive
                    #-----------------------------------------------------------
                    linear = linear_list[batch,:,primitive]
                    q = linear[0:9]
                    q = q.view(3,3)
                    q = q.contiguous()

                    t = linear[9:12]
                    t = t.view(1,3).expand(y.size(2),-1,-1).permute(1,2,0)
                    t = t.contiguous()
                    #-----------------------------------------------------------
                #---------------------------------------------------------------

                #not allowing the network to modify the plan before the spatial
                #transformation (ie using simple plan)
                #---------------------------------------------------------------
                else:

                    y = rand_grid

                    #generate the roation matrix & bias wrt the primitive
                    #-----------------------------------------------------------
                    linear = linear_list[batch,:,primitive]
                    q = linear[0:6]
                    q = q.view(2,3)
                    q = q.contiguous()

                    t = linear[6:9]
                    t = t.view(1,3)
                    t = t.expand(y.size(2),-1,-1).permute(1,2,0)
                    t = t.contiguous()
                    #-----------------------------------------------------------
                #---------------------------------------------------------------


                #apply the transformation matrix and the bias
                #---------------------------------------------------------------
                yq  = torch.matmul(y.permute(0,2,1).contiguous(),q)
                yq  = yq.permute(0,2,1).contiguous()
                yqt = torch.add(yq,t)
                #---------------------------------------------------------------

                #combine all the predictions of the primitives of one sample
                #---------------------------------------------------------------
                out_primitive.append(yqt)
            out_batch.append(torch.cat(out_primitive,2).contiguous())
            #-------------------------------------------------------------------

        #combine all the predictions of the sample
        #-----------------------------------------------------------------------
        out_batch = torch.cat(out_batch,0).contiguous()
        #-----------------------------------------------------------------------

        #configuration and its probability
        #-----------------------------------------------------------------------
        existance_prob[existance_prob == 0] = 1
        configuration = out_batch.transpose(2,1).contiguous()

        if constant_rep:
            ones = Variable(torch.ones(x.size(0))).cuda().contiguous()
            configuration_probability = ones
        else:
            configuration_probability = torch.prod(existance_prob,1)
        #-----------------------------------------------------------------------

        return configuration, configuration_probability, points_per_primitive




#BASELINE
class PointDecoder(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size/2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size/4)
        self.fc1 = nn.Linear(self.bottleneck_size, bottleneck_size)
        self.fc2 = nn.Linear(self.bottleneck_size, bottleneck_size/2)
        self.fc3 = nn.Linear(bottleneck_size/2, bottleneck_size/4)
        self.fc4 = nn.Linear(bottleneck_size/4, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points).transpose(1,2).contiguous()
        return x



class PointDecoderNormal(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(PointDecoderNormal, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size/2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size/4)
        self.fc1 = nn.Linear(self.bottleneck_size, bottleneck_size)
        self.fc2 = nn.Linear(self.bottleneck_size, bottleneck_size/2)
        self.fc3 = nn.Linear(bottleneck_size/2, bottleneck_size/4)
        self.fc4 = nn.Linear(bottleneck_size/4, self.num_points * 6)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 6, self.num_points).transpose(1,2).contiguous()
        return x


class AE_Baseline(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(AE_Baseline, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = PointDecoder(num_points = num_points, bottleneck_size = self.bottleneck_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AE_Baseline_normal(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(AE_Baseline_normal, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.encoder = nn.Sequential(
        PointNetfeatNormal(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = PointDecoderNormal(num_points = num_points, bottleneck_size = self.bottleneck_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SVR_Baseline(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, pretrained_encoder=False):
        super(SVR_Baseline, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder,  num_classes=bottleneck_size)
        self.decoder = PointDecoder(num_points = num_points, bottleneck_size = self.bottleneck_size)

    def forward(self, x):
        x = x[:,:3,:,:].contiguous()

        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    # print('testing our method...')
    # sim_data = Variable(torch.rand(1, 3, 400, 400))
    # model = PointNetAE_RNN_grid2mesh()
    # model.cuda()
    # out = model(sim_data.cuda())
    # print(out.size())

    # print('testing baseline...')
    # sim_data = Variable(torch.rand(1, 3, 400, 400))
    # model = PointNetAEBottleneck()
    # model.cuda()
    # out = model(sim_data.cuda())
    # print(out.size())

    print('testing PointSenGet...')
    sim_data = Variable(torch.rand(1, 4, 192, 256))
    model = Hourglass()
    # model.cuda()
    # out = model(sim_data.cuda())
    out = model(sim_data)
    print(out.size())

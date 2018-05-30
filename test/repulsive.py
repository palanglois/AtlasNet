from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Variable

def stitching_loss(input,target,alpha=0.1,beta=1):

    #getting size of the input/target
    #---------------------------------------------------------------------------
    sp = input.size(0)
    st = target.size(0)
    #---------------------------------------------------------------------------

    #copy the input and the target
    #---------------------------------------------------------------------------
    p = input.clone().contiguous()
    t = target.clone().contiguous()
    #---------------------------------------------------------------------------

    #expend the copy in order to apply matrix computation
    #---------------------------------------------------------------------------
    p = p.unsqueeze(2)
    p = p.expand(-1,-1,st).contiguous()
    p = p.permute(0,2,1).contiguous()
    t = t.expand(sp,-1,-1).contiguous()
    #---------------------------------------------------------------------------

    #can be skiped but usefull to display the result
    #---------------------------------------------------------------------------
    t = t.view(sp*st,3)
    p = p.view(sp*st,3)
    #---------------------------------------------------------------------------

    print(p)
    print(t)

    #compute the distance and using the exponnential to minimize the impact
    #of the farthest points
    #we had all the distance
    #---------------------------------------------------------------------------
    dist = torch.sum((p - t)**2,1)
    print(dist,alpha,beta)
    dist = beta*torch.exp(-0.5*(dist/alpha)**2)
    print(dist)
    dist = torch.sum(dist,0)
    #---------------------------------------------------------------------------

    return dist

def primitive_analysis(input,points_repartition):

    print(input)
    print(points_repartition)


    for i in range(points_repartition.size(0)):

        #-----------------------------------------------------------------------
        current = 0
        batch   = input[i,...].clone()
        #-----------------------------------------------------------------------

        for j in range(points_repartition.size(1)):

            print(batch)

            #split the primitive from the rest of the points
            #-------------------------------------------------------------------
            L = points_repartition[i,j]
            temp = batch[0:L,...]
            batch[0:L,...]  = batch[current:current+L,...]
            batch[current:current+L,...] = temp
            primitive_focus = batch[0:L,...].clone()
            other_primitive = batch[current+L:,...].clone()
            #-------------------------------------------------------------------

            #update the pointer to our current position in the batch
            #-------------------------------------------------------------------
            current = current + L
            #-------------------------------------------------------------------

            print(primitive_focus)
            print(other_primitive)
            #compute the stiching loss between the primitive and the rest of the
            #points
            #-------------------------------------------------------------------
            loss = stitching_loss(primitive_focus,other_primitive)
            #-------------------------------------------------------------------
            print(loss)
            exit(0)

sb = 4
sp = 10
st = 30

pprediction = Variable(torch.cuda.FloatTensor(sb,sp,3))
ttarget_obj = Variable(torch.cuda.FloatTensor(sb,st,3))

ppoint_per_primitives = Variable(torch.IntTensor(((4,2,2,1,1),(2,2,2,2,2),(2,2,2,2,2),(2,2,2,2,2)))).cuda()
# ppoint_per_primitives = Variable(torch.IntTensor(((1,1,1,1,1,2,3,5,5,5),(1,1,1,1,1,2,3,5,5,5),(1,1,1,1,1,2,3,5,5,5),(1,1,1,1,1,2,3,5,5,5)))).cuda()
pprediction.data.uniform_(-1,1)
ttarget_obj.data.uniform_(-1,1)

primitive_analysis(pprediction,ppoint_per_primitives)
L = stitching_loss(input=pprediction,target=ttarget_obj,alpha=0.1, beta=1)

print("the stitching loss is equal to : ", L)

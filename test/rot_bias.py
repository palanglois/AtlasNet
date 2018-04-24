import torch


def mult_bias(x,k,b):
    
    y = x.permute(0,2,1).contiguous()    
    y = torch.matmul(y,k).permute(0,2,1).contiguous()
    b = b.expand(x.size(2),-1,-1).permute(1,2,0).contiguous()
    return torch.add(y,b)

x = torch.Tensor(4,3,2).uniform_(0, 1).contiguous()
b = torch.Tensor(4,3).uniform_(0, 1).contiguous()
k = torch.Tensor(4,3,3).uniform_(0, 1).contiguous()

print "************* Input *************"
print x
print "********** Rot. Matrix **********"
print k
print "************* Bias **************"
print b
print "************ Result *************"
print mult_bias(x,k,b)


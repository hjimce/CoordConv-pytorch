#coding=utf-8
import numpy as np
from torch import nn
import torch
class CoordConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True):
        super(CoordConv,self).__init__()
        self.xychannel=None
        self.conv=nn.Conv2d(in_channels+2, out_channels, kernel_size, stride,padding=padding,dilation=dilation,groups=groups,bias=bias)



    def create_xychannel(self,height,width):
        xrange=np.arange(height).reshape(height,1)
        xones=np.ones((1,width))

        xone_map=(xrange*xones)/float(height-1)*2-1#记住归一化
        xchannel=np.expand_dims(np.expand_dims(xone_map, 0),0)



        yrange=np.arange(width).reshape(width,1)
        yones=np.ones((1,height))
        yone_map=(yrange*yones).transpose((1,0))/float(width-1)*2-1
        ychannel = np.expand_dims(np.expand_dims(yone_map, 0), 0)

        self.xychannel=torch.tensor(np.concatenate((xchannel,ychannel),axis=1),dtype=torch.float)
    def forward(self,inputs):
        batch,nchannel,height,width=inputs.shape
        if self.xychannel is None:
            self.create_xychannel(height,width)#xychannel.shape:(1,2,height,width) 
        batch_xychannels=self.xychannel.repeat((batch,1,1,1))
        if "cuda" in str(inputs.type()):
            batch_xychannels=batch_xychannels.cuda()
        batch_coordconv_inputs=torch.cat([batch_xychannels,inputs],dim=1)
        
        return self.conv(batch_coordconv_inputs) 
       


def test_coordconv():
    cv=CoordConv(3,8,3)
    inputs=torch.rand((2,3,4,5))
    print cv(inputs).shape

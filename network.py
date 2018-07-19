#coding=utf-8
# 这边随便写了一个网络模型，用于实验坐标卷积层和普通卷积层在效果上的差异
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import  Variable 
from coordconv import CoordConv
import matplotlib.pyplot as plt
import random
channel=8
class MyCoordConvNet(nn.Module):
    
    def __init__(self):
        super(MyCoordConvNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                CoordConv(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                CoordConv(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.model = nn.Sequential(
            conv_bn(  1,  channel-2, 2), #为了做更好的比较，假设两个网络的参数差不多，所以减去2
            conv_dw( channel-2,  channel*2-2, 2),
            conv_dw(channel*2-2, channel*4-2, 2),
            conv_dw(channel*4-2, channel*8-2, 2),
            nn.AdaptiveMaxPool2d((1,1))
        )
        self.fc = nn.Linear(channel*8-2, 30)

    def forward(self, x):
        x = self.model(x*2-1)#归一化处理
        x = x.view(-1,  channel*8-2)
        x = self.fc(x)
        return x
class MyConvNet(nn.Module):
    
    def __init__(self):
        super(MyConvNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.model = nn.Sequential(
            conv_bn(  1,  channel, 2), 
            conv_dw( channel,  channel*2, 2),
            conv_dw(channel*2, channel*4, 2),
            conv_dw(channel*4, channel*8, 2),
            nn.AdaptiveMaxPool2d((1,1))
        )
        self.fc = nn.Linear(channel*8, 30)

    def forward(self, x):
        x = self.model(x*2-1)#归一化处理
        x = x.view(-1,  channel*8)
        x = self.fc(x)
        return x
def mask_nan(predict,labels):
    mask=torch.isnan(labels)^1
    predict = torch.masked_select(predict, mask)
    labels = torch.masked_select(labels, mask)

    
    return predict,labels
def evaluate_model(loader, model, loss_fn, use_gpu = False):
    
    total_loss = 0
    for i, ( inputs, labels) in enumerate(loader):     
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
                
        # forward pass
        outputs = model(inputs)
        
        outputs, labels=mask_nan(outputs, labels)
        loss = loss_fn(outputs, labels)
        
        # metrics
        total_loss += loss.data[0]
            
    return (total_loss / i)

def test_predict(model, dset_loaders, use_gpu = False):
    reslut=[]
    for ii, (image_ori, labels) in enumerate(dset_loaders):
        
        if use_gpu:
            image = Variable(image_ori.cuda())
        else:
            image = Variable(image)
        outputs = model(image)
        outputs=outputs.data.cpu()
        for img,pre in zip(image_ori.numpy(),outputs.numpy()):
            reslut.append((img,pre))
    return reslut

        

def plot_face_Keypoints(result):
    
    for i,(x,y) in enumerate(result):
        plt.imshow(np.squeeze(x), cmap = 'gray')
        cordinates = y* 48 + 48
        plt.scatter(cordinates[::2], cordinates[1::2], marker='o', c='b', s=10)
        plt.savefig("result/%d.png"%i)
        plt.close()


def write_loss(loss,name="loss.txt"):
    with open(name,'w') as f:
        for l in loss:
            f.write(str(l)+' ')
def draw_loss(pathes):
    losses=[]
    for path1 in pathes:
        loss1=[]
        with open(path1,'r') as f:
            for l in f.readlines():
                p=l.split(" ")[:-1]
                for k in p:
                    loss1.append(float(k))
        losses.append(loss1)
    color=["r","b","g"]
    
    p1,=plt.plot(range(len(losses[0])),losses[0],color=color[0],label="Conv loss")
    p2,=plt.plot(range(len(losses[1])),losses[1],color=color[1],label="CoordConv loss")

    
    plt.legend([p1,p2], ["Conv loss","CoordConv loss"])

    plt.show()

            

def train(model, train_loader, test_loader ,num_epochs, loss_fn, optimizer ):
    
    loss_train = []
    loss_test = []


    for epoch in range(num_epochs):
        model.train()
        loss_cpu=0
        for i, (inputs, labels) in enumerate(train_loader):

            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            predict = model(inputs)
            
            predict, labels=mask_nan(predict, labels)
            loss = loss_fn(predict, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_cpu+=loss.data.cpu().numpy()*10000
        if epoch%5!=0:
            continue
        model.eval()
        loss_teste=evaluate_model(test_loader, model,loss_fn, "True").data.cpu().numpy()*10000

        loss_train.append(loss_cpu/i)
        loss_test.append(loss_teste)

        print "Epoch:",epoch,  "Train:", loss_cpu/i,  "Test:" ,loss_teste

    
    
    return loss_train, loss_test, model






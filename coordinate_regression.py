#coding=utf-8
import coordconv
import torch.nn
import face_key_point

import  matplotlib.pyplot as plt
import torchvision
import numpy as np
import network
import torch.optim as optim
import os
coordconvnet=True
#network.draw_loss(["train_conv.txt","train.txt"])
if coordconvnet:
        model = network.MyCoordConvNet().cuda()
else:
        model = network.MyConvNet().cuda()
def train(model):
        torch.manual_seed(2)
        landmarks=face_key_point.load_face_landmark_data(path=os.path.join("data","training.csv"),batch_size=32)
        train_d=landmarks.data["train"]
        valid_d=landmarks.data["valid"]
        loss_fn = torch.nn.MSELoss()
        optimizer =  optim.RMSprop(model.parameters(), lr=1e-3,weight_decay=1e-5)
        num_epochs =500
        params = {'model' : model, 
                'train_loader':train_d,
                'test_loader':valid_d,
                'num_epochs': num_epochs,
                'loss_fn': loss_fn,
                'optimizer': optimizer, 
                }


        loss_train, loss_test, model = network.train(**params)
        network.write_loss(loss_train,"train.txt")
        torch.save(model.state_dict(),'model.pt')
        





def test(model):
        landmarks=face_key_point.load_face_landmark_data(path=os.path.join("data","test.csv"),batch_size=32,is_test=True)
        train_d=landmarks.data["test"]
        print "restore model"
        model.load_state_dict(torch.load('model.pt'))
        model.eval()
        preds_test = network.test_predict(model, train_d, True)
        network.plot_face_Keypoints(preds_test)

train(model)
#test(model)









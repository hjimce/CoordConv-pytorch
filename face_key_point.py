import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import os
from torch.utils.data.sampler import SubsetRandomSampler

class FaceKeyPointsDataset(Dataset):

    def __init__(self, X, y, transforms=None):
        
        self.transform = transforms    
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        image = self.X[index]
        keypoints = self.y[index]        
        return image, keypoints

    def __len__(self):
        return len(self.X)
class load_face_landmark_data():
    def __init__(self,path,batch_size=32,is_test=False):
        X_train, y_train = self.get_data(path)
        dsets = {
            'train': FaceKeyPointsDataset(X_train, y_train),
            'valid':  FaceKeyPointsDataset(X_train, y_train)
            }
        
        num_train = len(dsets['train'])
        indices = list(range(num_train))
        split = int(np.floor(0.2* num_train))



        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self.data={}
        if is_test:
            self.data["test"] = DataLoader(FaceKeyPointsDataset(X_train, y_train), batch_size,shuffle=False)
        else:
            self.data["train"] = DataLoader(dsets["train"], batch_size, sampler=train_sampler)
            self.data["valid"] = DataLoader(dsets["valid"], batch_size, sampler=valid_sampler,shuffle=False)

    def get_data(self,path2data):

        df = pd.read_csv(path2data)
        #df = df.dropna()

        df['Image'] = df['Image'].apply(lambda img:  np.fromstring(img, sep = ' '))
        X = np.vstack(df['Image'].values)
        X = X / 255.   # scale pixel values to [0, 1]
        X = X.astype(np.float32)
        X = X.reshape(-1, 1, 96, 96)


        y = df[df.columns[:-1]].values
        y = (y - 48) / 48 
        
        y = y.astype(np.float32)


        return X, y






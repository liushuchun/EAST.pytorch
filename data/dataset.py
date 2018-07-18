import torch.utils.data as data
import os
import numpy as np
from torchvision import transforms as T
from PIL import Image


class ImageDataSet(data.Dataset):

    def __init__(self,root,transforms=None,train=True,test=False):


        self.test=test

        imgs=[os.path.join(root,img) for img in os.listdir(root)]

        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

        else:
            imgs=sorted(imgs,key=lambda  x:int(x.split('.')[-2]))

        imgs_num=len(imgs)

        #shuffle images

        np.random.seed(100)

        imgs=np.random.permutation(imgs)

        if self.test:
            self.imgs=imgs
        elif train:
            self.imgs=imgs[:int(0.7*imgs_num)]

        else:
            self.imgs=imgs[int(0.7*imgs_num)]


        if transforms is None:
            normalize=T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])




    def __getitem__(self, index):
        img_path=self.imgs[index]


        data=Image.open(img_path)
        label=""
        return data,label


    def __len__(self):
        return len(self.imgs)





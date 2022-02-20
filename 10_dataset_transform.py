#'''
#Transforms can be applied to PIL images, tensors, ndarrays, or custom dataduring creation of the DataSet complete list of built-in transforms:  https://pytorch.org/docs/stable/torchvision/transforms.html
########### !!!!!!!!!!!!On Images!!!!!!!!!!!!!
#---------
#CenterCrop, Grayscale, Pad, RandomAffine
#RandomCrop, RandomHorizontalFlip, RandomRotation
#Resize, Scale
##############################
#On Tensors
#----------
#LinearTransformation, Normalize, RandomErasing,Conversion
#----------
#ToPILImage: from tensor or ndrarray
#ToTensor : from numpy.ndarray or PILImage
#Generic
#-------
#Use Lambda 
#Custom
#------
#Write own class
#Compose multiple Transforms
#---------------------------
#composed = transforms.Compose([Rescale(256),RandomCrop(224)])
from genericpath import samefile
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np  

class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.transform = transform

    def __getitem__(self, index):
        #dataset[0]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        #len(dataset)
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MultTransform:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=ToTensor)
firts_data = dataset[0]
features, labels = firts_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MultTransform(2)])
dataset = WineDataset(transform=composed)
firts_data = dataset[0]
features, labels = firts_data
print(features)
print(type(features), type(labels))
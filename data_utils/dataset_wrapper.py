import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets

# our custom data transformers
from data_utils.gaussian_blur import GaussianBlur
from data_utils.eye_augmentation import CustomZoom
import pandas as pd
#dataset classe
from data_utils.eye_data import EyeDataset

np.random.seed(0)


class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)



    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        train_dataset = EyeDataset(csv_file='./raw_dataset/train.csv',root_dir='/hpc/tkal976/EyeKnowYouCropped', #change the roo dir
                                       transform=SimCLRDataTransform(data_augment))

        valid_dataset = EyeDataset(csv_file='./raw_dataset/valid.csv',root_dir='/hpc/tkal976/EyeKnowYouCropped', #change the root dir
                                       transform=SimCLRDataTransform(data_augment))

        train_loader = self.get_custom_data_loaders(train_dataset,'train')
        valid_loader = self.get_custom_data_loaders(valid_dataset,'valid')

   



        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),  #we can write cutom data augmentation classes
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])), #this is the only custom data augmentation method
                                              transforms.ToTensor()])
        return data_transforms

    def get_custom_data_loaders(self, custom_dataset,split='train'):
        # obtain data indices
        num_examples = len(custom_dataset)
        indices = list(range(num_examples))
        np.random.shuffle(indices)

        print('number of '+split+' examples:',num_examples)


    
        # pre-defined sampler for sample indices 
        data_sampler = SubsetRandomSampler(indices)

    
 
        data_loader = DataLoader(custom_dataset, batch_size=self.batch_size, sampler=data_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
 

        return data_loader




class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
     
        xi = self.transform(sample) #1st random transformation for the batch
        xj = self.transform(sample)  #2nd random transformation for the transoformations

    
        return xi, xj

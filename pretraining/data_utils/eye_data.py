import torch
import os
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
from PIL import Image
import numpy as np

class EyeDataset(Dataset): #subclass of pytorch Dataset
    """EyeKnowYou dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with eye images.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied  
                on a sample (please add your own transformations.).  (let's use cutom transform )
        """
        self.eye_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self): #give the size of the dataset
        return len(self.eye_frame)

    def __getitem__(self, idx):

        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.eye_frame.iloc[idx, 0])
        image = Image.open(img_name)#np.asarray(io.imread(img_name)) #make sure u use PIL, or a tensor

        # eye = self.eye_frame.iloc[idx, 1:]
        # eye = np.array([eye])
        # eye = eye.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'eye': eye}
        sample=image#{'image': image}

        if self.transform:
            
            sample = self.transform(sample)  #from here the code move to transform functiion

            #sample is a tuple which include two augmentations of the same example


       

           
        


        return sample
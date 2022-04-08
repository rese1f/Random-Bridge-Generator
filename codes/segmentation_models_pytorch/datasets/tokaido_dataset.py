import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd

class TokaidoDataset(Dataset):
    def __init__(
            self,
            root_dir, 
            map_dir,  
            augmentation=True, 
    ):  
        # read csv file
        # FORMAT
        #   image file name
        #   component label file name
        #   damage label file name
        #   depth image file name
        
        self.ids = pd.read_csv(map_dir, header=None)
        # choose only regular images with close-up damage
        self.ids = self.ids[(self.ids[5]==True)&(self.ids[6]==True)] 
        
        self.image_ids = [os.path.join(root_dir, image_id) for image_id in self.ids.iloc[:, 0]]
        self.cmp_ids = [os.path.join(root_dir, cmp_id) for cmp_id in self.ids.iloc[:, 1]]
        self.dmg_ids = [os.path.join(root_dir, dmg_id) for dmg_id in self.ids.iloc[:, 2]]
        self.depth_ids = [os.path.join(root_dir, depth_id) for depth_id in self.ids.iloc[:, 3]]
        self.augmentation = augmentation
    
    def __getitem__(self, i):
        
        sample = {}
        size = (640, 320)
        
        img = np.array(Image.open(self.image_ids[i]).convert("RGB").resize(size, 2))
        cmp = np.array(Image.open(self.cmp_ids[i]).resize(size, 0))
        dmg = np.array(Image.open(self.dmg_ids[i]).resize(size, 0))
        depth = np.array(Image.open(self.depth_ids[i]).resize(size, 2))
        
        if self.augmentation == True:
            pass
        
        # convert to one-hot
        # cmp, dmg = onehot(cmp, 8), onehot(dmg, 3)
        
        # convert to other format HWC -> CHW
        sample['img'] = np.moveaxis(img, -1, 0)
        # sample['cmp'] = np.moveaxis(cmp, -1, 0)
        # sample['dmg'] = np.moveaxis(dmg, -1, 0)
        sample['cmp'] = np.expand_dims(cmp, 0)
        sample['dmg'] = np.expand_dims(dmg, 0)
        sample['depth'] = np.expand_dims(depth, 0)
        
        return sample
        
    def __len__(self):
        return len(self.ids)

def onehot(label, N):
    return (np.arange(N) == label[...,None]-1).astype(int)
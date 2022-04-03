import os
import torch
from torch.utils.data import Dataset as BaseDataset
import torchvision.transforms as transforms
import numpy as np
import cv2
import pandas as pd

class TokaidoDataset(BaseDataset):
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
        
        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([320,640]),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
            ])
        
        img = cv2.imread(self.image_ids[i], flags=-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_transforms(img)
        
        cmp = cv2.imread(self.cmp_ids[i], flags=-1)
        cmp = cv2.resize(src = cmp,
                         dsize = (640,320), 
                         interpolation = cv2.INTER_LINEAR)
        cmp = torch.as_tensor(cmp)
        
        dmg = cv2.imread(self.dmg_ids[i], flags=-1)
        dmg = cv2.resize(src = dmg,
                         dsize = (640,320), 
                         interpolation = cv2.INTER_LINEAR)
        dmg = torch.as_tensor(dmg)
        
        depth = cv2.imread(self.depth_ids[i], flags=-1)
        depth = np.array(depth) / (2**16 - 1) * (30 - 0.5) + 0.5
        depth = torch.as_tensor(depth)
        
        # apply augmentations
        if self.augmentation:
            p = np.random.choice([0, 1])
            aug_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p)
            ])
            img = aug_transforms(img)
            cmp = aug_transforms(cmp)
            dmg = aug_transforms(dmg)
            depth = aug_transforms(depth)
            
        # return img, cmp, dmg, depth, self.image_ids[i]
        return img, cmp
        
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def onehot(label, N):
        pass
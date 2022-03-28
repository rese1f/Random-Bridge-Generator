import os
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from segmentation_models_pytorch.datasets import TokaidoDataset

if __name__ == '__main__':

    comp_dataset = TokaidoDataset(
        images_dir = r'C:/Users/Reself/Downloads/Tokaido/img_syn_raw/train/',
        masks_dir =  r'C:/Users/Reself/Downloads/Tokaido/synthetic/train/labcmp/',
        classes = ['nonbridge', 
                'slab', 
                'beam', 
                'column', 
                'nonstructural components', 
                'rail', 
                'sleeper', 
                'others'],
        augmentation = True,
        preprocessing = True,
    )

    comp_dataset.__getitem__(0)
import os
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from segmentation_models_pytorch.datasets import TokaidoDataset
from segmentation_models_pytorch.utils import visualize

import pdb

if __name__ == '__main__':

    train_dataset = TokaidoDataset(
        map_dir = r'C:/Users/Reself/Downloads/Tokaido/files_train.csv',
        root_dir =  r'C:/Users/Reself/Downloads/Tokaido/',
        augmentation = True,
    )
    
    # model = smp.FPN(
    #         encoder_name='resnet34', 
    #         encoder_weights=None,
    #         in_channels=3, 
    #         classes=8, 
    #         activation='sigmoid',
    # )
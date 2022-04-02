import os
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from configs import parse_args

import pdb

if __name__ == '__main__':
    
    args = parse_args()
    print(args)
    
    train_dataset = smp.datasets.TokaidoDataset(
        map_dir = r'C:/Users/Reself/Downloads/Tokaido/files_train.csv',
        root_dir =  r'C:/Users/Reself/Downloads/Tokaido/',
        augmentation = args.aug,
    )

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=os.cpu_count())  # type: ignore
    
    model = smp.FPN(
            encoder_name=args.model, 
            encoder_weights=None,
            in_channels=3, 
            classes=8, 
            activation='sigmoid',
    )
    
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=args.learning_rate),
    ])
    
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=args.device,
        verbose=True,
    )
    
    for i in range(args.epoch):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
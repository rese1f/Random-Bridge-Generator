import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.datasets.spine_dataset import Spine_Dataset

from torch.utils.data import DataLoader, random_split
from configs import parse_args
import matplotlib.pyplot as plt
from train import YasuoModel


def load_model(ckpt_path, in_channel, out_classes, args):
    file_name = os.listdir(ckpt_path)
    file_path = os.path.join(ckpt_path, file_name[0])

    # 1. create dataset
    dataset = smp.datasets.TokaidoDataset(
        map_dir="files_train.csv",
        root_dir="/mnt/sdb/Tokaido_dataset/",
        augmentation=args.aug,
        to_one_hot=args.to_one_hot,
    )

    # 2. split dataset
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. generate dataloader
    loader_args = dict(batch_size=args.batch_size, num_workers=16, pin_memory=True)
    train_dataloader = DataLoader(dataset=train_set, shuffle=True, **loader_args)  # type: ignore os.cpu_count()
    val_dataloader = DataLoader(dataset=val_set, shuffle=False, **loader_args)

    # 4. create a model
    model = YasuoModel(
        arch=args.arch,
        encoder_name=args.backbone,
        encoder_weights=None,
        in_channels=in_channel,
        out_classes=out_classes,
        lr=args.learning_rate,
        to_one_hot=args.to_one_hot,
    )

    # 5. load ckpt file
    model = YasuoModel.load_from_checkpoint(
        file_path,
        arch=args.arch,
        encoder_name=args.backbone,
        encoder_weights=None,
        in_channels=in_channel,
        out_classes=out_classes,
        lr=args.learning_rate,
        to_one_hot=args.to_one_hot,
    )

    # 6. input the img
    batch = next(iter(val_dataloader))
    for i in range(3):
        batch = next(iter(train_dataloader))
        
    with torch.no_grad():
        model.eval()
        output = model(batch["img"])
    pr_masks = torch.argmax(output, dim=1)

    return batch, pr_masks


def vis_save(batch, pr_masks, save_path, to_one_hot):
    cnt = 0
    for image, gt_mask, pr_mask in zip(batch["img"], batch["dmg"], pr_masks):
        cnt = cnt + 1

        if to_one_hot:
            gt_mask = torch.argmax(gt_mask, dim=0) / 7 * 255
        else:
            gt_mask = gt_mask / 2 * 255

        pr_mask = pr_mask / 2 * 255

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy())  # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy())  # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")
        plt.savefig("./predicted_results/compare{}.png".format(cnt))
        plt.show()


if __name__ == "__main__":

    ckpt_path = "./lightning_logs/version_94/checkpoints/"
    save_path = "./compare.png"

    args = parse_args()
    batch, pr_masks = load_model(ckpt_path, 3, 3 , args)
    vis_save(batch, pr_masks, save_path, to_one_hot=args.to_one_hot)


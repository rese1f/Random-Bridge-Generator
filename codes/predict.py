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


def load_model(ckpt_path, in_channel, out_classes, args, do_seg):
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
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. generate dataloader
    loader_args = dict(batch_size=args.batch_size,
                       num_workers=16, pin_memory=True)
    # type: ignore os.cpu_count()
    train_dataloader = DataLoader(
        dataset=train_set, shuffle=True, **loader_args)
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
    for _ in range(2):
        batch = next(iter(train_dataloader))

    with torch.no_grad():
        model.eval()
        preds = model(batch["img"])
    if do_seg:
        preds = torch.argmax(preds, dim=1)

    return batch, preds


def vis_save(batch, preds, save_path, do_seg, to_one_hot):
    cnt = 0
    for image, gt, pred in zip(batch["img"], batch["depth"], preds):
        cnt = cnt + 1

        if do_seg:
            gt = torch.argmax(gt, dim=0) / 2 * \
                255 if to_one_hot else gt / 2 * 255
            pred = pred / 2 * 255
        else:
            gt = gt * 255
            pred = pred * 255

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        # just squeeze classes dim, because we have only one class
        plt.imshow(gt.numpy())
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        # just squeeze classes dim, because we have only one class
        plt.imshow(pred.squeeze().numpy())
        plt.title("Prediction")
        plt.axis("off")
        plt.savefig(f"./predicted_results/compare{cnt}.png")
        plt.show()


if __name__ == "__main__":

    # ckpt_path = "./lightning_logs/version_223/checkpoints/"
    ckpt_path = "./lightning_logs/version_241/checkpoints/"
    save_path = "./compare.png"
    do_seg = False  # decide whether to do segment or not

    args = parse_args()
    batch, preds = load_model(ckpt_path, 3, 1, args, do_seg)
    vis_save(batch, preds, save_path, do_seg, to_one_hot=args.to_one_hot)

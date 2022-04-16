import os
import numpy as np
from pickletools import optimize
import torch
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader, random_split
from dice_score import dice_loss
from configs import parse_args


class YasuoModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, lr, to_one_hot, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            activation="softmax",
            **kwargs,
        )
        # preprocessing parameteres for image
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.to_one_hot = to_one_hot
        self.lr = lr
        # for image segmentation dice loss could be the best first choice
        # self.loss_fn = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True)
        self.loss_fn = dice_loss() if self.to_one_hot else smp.losses.SoftCrossEntropyLoss(smooth_factor=0.01)

    def forward(self, img):
        output = self.model(img)
        return output

    def shared_step(self, batch, stage):

        img = batch["img"]
        assert img.ndim == 4

        h, w = img.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        gt = batch["cmp"]
        if self.to_one_hot:
            assert gt.ndim == 4
        else:
            assert gt.ndim == 3

        output = self.forward(img)
        loss = self.loss_fn(output, gt.long())

        pred_mask = torch.argmax(output, dim=1)
        if self.to_one_hot:
            gt = torch.argmax(gt, dim=1)

        # pred_mask and gt must be (N,H,W)
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, gt, mode="multiclass", num_classes=8)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # store the mIou for each class
        with open("./miou.txt", "ab") as f:
            per_class_iou = torch.mean(smp.metrics.iou_score(tp, fp, fn, tn, reduction=None), dim=0)
            per_class_iou = np.array(per_class_iou).reshape(1, 8)
            np.savetxt(f, per_class_iou, delimiter=" ")

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=250, T_mult=2, eta_min=0, last_epoch=-1
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":

    args = parse_args()
    print(args)

    # 1. create a dataset
    dataset = smp.datasets.TokaidoDataset(
        map_dir="files_train.csv",
        root_dir="/mnt/sdb/Tokaido_dataset/",
        augmentation=args.aug,
        to_one_hot=args.to_one_hot,
    )

    # 2. split the dataset
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. generate dataloaders
    loader_args = dict(batch_size=args.batch_size, num_workers=16, pin_memory=True)
    train_dataloader = DataLoader(dataset=train_set, shuffle=True, **loader_args)  # type: ignore os.cpu_count()
    val_dataloader = DataLoader(dataset=val_set, shuffle=False, **loader_args)

    # 4. create a model
    model = YasuoModel(
        arch=args.arch,
        encoder_name=args.backbone,
        encoder_weights="imagenet",
        in_channels=3,
        out_classes=8,
        lr=args.learning_rate,
        to_one_hot=args.to_one_hot
    )

    # 5. define a trainer
    trainer = pl.Trainer(gpus=1, max_epochs=args.num_epoch,)

    # 6. train the network
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

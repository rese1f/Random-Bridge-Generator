import os
import numpy as np
from pickletools import optimize
import torch
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader, random_split
from custom_loss.depth_differ import depth_loss
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
            activation=None,
            **kwargs,
        )
        # preprocessing parameteres for image
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.classes = out_classes
        self.to_one_hot = to_one_hot
        self.lr = lr

        # self.loss_fn = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True)
        # self.loss_fn = dice_loss() if self.to_one_hot else smp.losses.SoftCrossEntropyLoss(smooth_factor=0.01, ignore_index=255)
        self.loss_fn = depth_loss()
        self.loss_fn1 = smp.losses.DiceLoss(
            mode="multiclass", from_logits=False)
        self.loss_fn2 = smp.losses.FocalLoss(
            mode="multiclass", alpha=0.8, gamma=2.0)

    def forward(self, img):
        output = self.model(img)
        return output

    def shared_step(self, batch, stage):
        img = batch["img"]
        assert img.ndim == 4

        h, w = img.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # choose the specific kind of pics to train
        # gt = batch["cmp"]
        gt = batch["depth"]

        if self.to_one_hot:
            assert gt.ndim == 4
            gt = gt.to(torch.float32)
        else:
            assert gt.ndim == 3

        # prepare for bce_loss
        # w = torch.tensor([0.01, 1.5, 0.1]) # weight of each label
        # gt_ = torch.argmax(gt, dim=1, keepdims=True).to(torch.int64)
        # weight = (gt_ == 0) * w[0] + (gt_ == 1) * w[1] + (gt_ == 2) * w[2]
        # weight = weight.repeat(1, 3, 1, 1)

        # already activated
        output = self.forward(img)  # (b, 1, h, w)
        # import pdb;pdb.set_trace()
        loss = self.loss_fn(output, gt)

        # b, h, w = output.shape
        # output = output.reshape(b, -1)
        # min_depth = torch.min(output, dim=1)[0].reshape(b, -1)
        # max_depth = torch.max(output, dim=1)[0].reshape(b, -1)
        # output = (output - min_depth) / (max_depth - min_depth)
        # output = (output * 29.5 + 0.5).reshape(b, h, w)

        # calculate loss
        # loss = self.loss_fn(output, gt)
        # loss =  10 * F.binary_cross_entropy(output, gt, weight=weight)
        # loss = 1.2 * self.loss_fn1(output, gt_.squeeze(1)) + 1 * self.loss_fn2(output, gt_.squeeze(1))
        # loss = 0.7 * self.loss_fn1(output, gt) + 0.3 * self.loss_fn2(output, gt)
        # loss = self.loss_fn1(output, gt)

        # pred_mask = torch.argmax(output, dim=1)

        # calculate accs
        output = output.squeeze()
        threshold = [0.01, 0.02, 0.05, 0.1]
        pred_mask1 = (torch.abs(output - gt) <= threshold[0]).long()
        pred_mask2 = (torch.abs(output - gt) <= threshold[1]).long()
        pred_mask3 = (torch.abs(output - gt) <= threshold[2]).long()
        pred_mask4 = (torch.abs(output - gt) <= threshold[3]).long()
        gt = torch.ones(*gt.shape)
        
        assert gt.size() == pred_mask1.size()
        
        acc1 = torch.sum(pred_mask1) / torch.sum(gt)
        acc2 = torch.sum(pred_mask2) / torch.sum(gt)
        acc3 = torch.sum(pred_mask3) / torch.sum(gt)
        acc4 = torch.sum(pred_mask4) / torch.sum(gt)
        acc = [acc1, acc2, acc3, acc4]

        if self.to_one_hot:
            gt = torch.argmax(gt, dim=1)

        # pred_mask and gt must be (N,H,W)
        # tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, gt, mode="multiclass", num_classes=2)

        return {
            "loss": loss,
            "acc": acc,
            # "tp": tp,
            # "fp": fp,
            # "fn": fn,
            # "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        # tp = torch.cat([x["tp"] for x in outputs])
        # fp = torch.cat([x["fp"] for x in outputs])
        # fn = torch.cat([x["fn"] for x in outputs])
        # tn = torch.cat([x["tn"] for x in outputs])
        acc = [x["acc"] for x in outputs]
        acc = torch.Tensor(acc)
        acc = torch.mean(acc, dim=0)

        # # per image IoU means that we first calculate IoU score for each image
        # # and then compute mean over these scores
        # per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # # dataset IoU means that we aggregate intersection and union over whole dataset
        # # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # # in this particular case will not be much, however for dataset
        # # with "empty" images (images without target class) a large gap could be observed.
        # # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        # dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # # store the mIou for each class
        # with open("./dmg-" + args.arch + args.backbone + ".txt", "ab") as f:
        #     per_class_iou = torch.mean(smp.metrics.iou_score(tp, fp, fn, tn, reduction=None), dim=0).tolist()
        #     per_class_iou.append(dataset_iou.item())
        #     per_class_iou = np.array(per_class_iou).reshape(1, self.classes + 1)
        #     np.savetxt(f, per_class_iou, delimiter=",")

        metrics = {
            #     # f"{stage}_per_image_iou": per_image_iou,
            #     f"{stage}bgd_iou": per_class_iou[0, 0],
            #     f"{stage}small_leak_iou": per_class_iou[0, 1],
            #     f"{stage}big_leak_iou": per_class_iou[0, 2],
            #     f"{stage}_dataset_iou": dataset_iou,
            f"{stage}acc.01": acc[0],
            f"{stage}acc.02": acc[1],
            f"{stage}acc.05": acc[2],
            f"{stage}acc.1": acc[3],
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
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-8)
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
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. generate dataloaders
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
        encoder_weights="imagenet",
        in_channels=3,
        out_classes=1,
        lr=args.learning_rate,
        to_one_hot=args.to_one_hot,
    )

    # 5. define a trainer
    trainer = pl.Trainer(gpus=1, max_epochs=args.num_epoch)

    # 6. train the network
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

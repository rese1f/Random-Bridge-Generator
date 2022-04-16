import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.datasets.spine_dataset import Spine_Dataset

from torch.utils.data import DataLoader, random_split
from configs import parse_args
from dice_score import dice_loss
import matplotlib.pyplot as plt


class YoneModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, lr, threshold=0.5, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            activation="sigmoid",
            **kwargs,
        )
        # preprocessing parameteres for image
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        # self.loss_fn = smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE, alpha=0.75)
        # self.loss_fn = smp.losses.MCCLoss()
        # self.loss_fn = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)
        self.loss_fn = dice_loss() 
        
        self.threshold = threshold
        self.lr = lr

    def forward(self, img):
        # img has already normalized in dataset module!
        output = self.model(img)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=250, T_mult=2, eta_min=0, last_epoch=-1
        )
        return [optimizer], [scheduler]

    def shared_step(self, batch, stage):

        img = batch["img"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert img.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = img.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        gt = batch["mask"]
        assert gt.ndim == 4

        output = self.forward(img)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(output, gt)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding

        # prob_mask = output.sigmoid()
        pred_mask = (output > self.threshold).type(torch.uint8)
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), gt.long(), mode="binary")

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

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores

        # per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.

        # dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        rec = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        prec = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, "micro-imagewise")

        metrics = {
            # f"{stage}_per_image_iou": per_image_iou,
            # f"{stage}_dataset_iou": dataset_iou,
            f"{stage}rec": rec,
            f"{stage}prec": prec,
            f"{stage}f1": f1,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        # return self.shared_epoch_end(outputs, "train")
        pass

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")


if __name__ == "__main__":

    args = parse_args()
    print(args)
    # 1. create dataset
    dataset = Spine_Dataset(
        images_dir="/home/pose3d/projs/UNet_Spine_Proj/UNet_Spine/data/imgs",
        masks_dir="/home/pose3d/projs/UNet_Spine_Proj/UNet_Spine/data/masks",
        augmentation=args.aug,
    )

    # 2. split dataset
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. generate dataloader
    loader_args = dict(batch_size=args.batch_size, num_workers=16, pin_memory=True)
    train_dataloader = DataLoader(dataset=train_set, shuffle=True, **loader_args)  # type: ignore os.cpu_count()
    val_dataloader = DataLoader(dataset=val_set, shuffle=False, **loader_args)

    model = YoneModel(
        arch=args.arch,
        encoder_name=args.backbone,
        encoder_weights=None,
        in_channels=1,
        out_classes=1,
        lr=args.learning_rate,
    )

    trainer = pl.Trainer(gpus=1, max_epochs=args.num_epoch)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

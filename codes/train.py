import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from configs import parse_args

class YasuoModel(pl.LightningModule):
    
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, lr, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, 
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels, 
            classes=out_classes, 
            activation='sigmoid',
            **kwargs
        )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, 
                                           from_logits=True,
                                           ignore_index=-1)
        self.lr = lr

    def forward(self, img):
        # normalize image here
        img = (img - self.mean) / self.std
        output = self.model(img)
        return output

    def shared_step(self, batch, stage):
        
        img = batch['img']
        
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
        
        gt = batch['cmp']
        assert gt.ndim == 4
        
        output = self.forward(img)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(output, gt)
        
        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        _, pred_mask = torch.max(output.sigmoid(),dim=1,keepdim=True)
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), gt.long(), mode="multiclass", num_classes=8, ignore_index=-1)

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

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
    
if __name__ == '__main__':
    
    args = parse_args()
    print(args)
    
    train_dataset = smp.datasets.TokaidoDataset(
        map_dir = args.map_dir,
        root_dir =  args.root_dir,
        augmentation = args.aug,
    )

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=0)  # type: ignore os.cpu_count()
    
    model = YasuoModel(arch=args.arch, 
                       encoder_name=args.backbone,
                       encoder_weights=None,
                       in_channels=3,
                       out_classes=8,
                       lr=args.learning_rate)
    
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=args.num_epoch,
    )

    trainer.fit(
        model, 
        train_dataloaders=train_dataloader,
    )
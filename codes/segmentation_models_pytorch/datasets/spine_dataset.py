import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .augment import Compose, RandomFlip_LR, RandomFlip_UD, RandomRotate


class Spine_Dataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, augmentation: bool = True, scale: float = 1.0):
        self.augmentation = augmentation
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.scale = scale
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith(".")]
        if not self.ids:
            raise RuntimeError(f"No input file found in {images_dir}, make sure you put your images there")
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        # img.shape (1, w, h)  mask.shape (1, w, h)
        img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in [".npz", ".npy"]:
            return Image.fromarray(np.load(filename))
        elif ext in [".pt", ".pth"]:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)  # return (w, h)

    @staticmethod
    def transform(img, mask):
        data_transforms = Compose([RandomFlip_LR(prob=0.5), RandomFlip_UD(prob=0.5), RandomRotate()])
        return data_transforms(img, mask)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(f"{name}.*"))
        img_file = list(self.images_dir.glob(f"{name}.*"))


        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        img = torch.as_tensor(img.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()
        if self.augmentation:
            img, mask = self.transform(img, mask)

        return {"img": img, "mask": mask}

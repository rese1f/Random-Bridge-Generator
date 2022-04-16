import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd


class TokaidoDataset(Dataset):
    def __init__(self, root_dir, map_dir, augmentation, to_one_hot):
        # read csv file
        # FORMAT
        #   image file name
        #   component label file name
        #   damage label file name
        #   depth image file name
        self.ids = pd.read_csv(map_dir, header=None)
        # choose only regular images with close-up damage
        self.ids = self.ids[(self.ids[5] == True) & (self.ids[6] == True)]

        self.image_ids = [os.path.join(root_dir, image_id.replace("\\", "/")) for image_id in self.ids.iloc[:, 0]]
        self.cmp_ids = [os.path.join(root_dir, cmp_id.replace("\\", "/")) for cmp_id in self.ids.iloc[:, 1]]
        # self.dmg_ids = [os.path.join(root_dir, dmg_id.replace("\\","/")) for dmg_id in self.ids.iloc[:, 2]]
        # self.depth_ids = [os.path.join(root_dir, depth_id.replace("\\","/")) for depth_id in self.ids.iloc[:, 3]]
        self.augmentation = augmentation
        self.to_one_hot = to_one_hot

    def __getitem__(self, i):

        sample = {}
        size = (640, 320)
        img = np.array(Image.open(self.image_ids[i]).convert("RGB").resize(size, 2))
        cmp = np.array(Image.open(self.cmp_ids[i]).resize(size, 0)) - 1

        # dmg = np.array(Image.open(self.dmg_ids[i]).resize(size, 0))-1
        # depth = np.array(Image.open(self.depth_ids[i]).resize(size, 2))

        if self.augmentation == True:
            pass

        # convert to one-hot
        if self.to_one_hot:
            cmp = onehot(cmp, 8)
            cmp = np.transpose(cmp, (2, 0, 1))

        # convert to other format HWC -> CHW
        img = np.transpose(img, (2, 0, 1)) / 255

        # transfer to tensor
        img = torch.as_tensor(img.copy()).float().contiguous()
        cmp = torch.as_tensor(cmp.copy()).long().contiguous()

        sample["img"] = img
        sample["cmp"] = cmp

        return sample

    def __len__(self):
        return len(self.ids)


def onehot(label, N):
    return (np.arange(N) == label[..., None] - 1).astype(int)


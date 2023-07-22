import os
import math
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class LocalizeDataset(Dataset):
    def __init__(self, src_dir, train=True, transform=None, target_size=128, **kwargs):

        super().__init__(**kwargs)
        self.cam_paths = []
        self.x = []
        self.y = []
        self.transform = transform
        self.target_size = target_size

        csv_f = os.path.join(src_dir, "meta_data.csv")
        all_meta = pd.read_csv(csv_f, index_col=0)
        meta_x = all_meta[['x']]
        meta_y = all_meta[['y']]
        files = os.listdir(src_dir)
        cam_pic_files = [f for f in files if '_camera.png' in f]

        idxs = []
        for cam_f in cam_pic_files:
            idx = cam_f.replace('_camera.png', '').split('_')[-1]
            idxs.append(int(idx))
        idxs = sorted(idxs)

        for idx in idxs:
            cam_p = os.path.join(src_dir, "%d_camera.png" % idx)
            self.cam_paths.append(cam_p)
            x = meta_x.loc[idx].to_numpy()
            y = meta_y.loc[idx].to_numpy()
            self.x.append(x)
            self.y.append(y)

        self.cam_paths = np.array(self.cam_paths)
        self.x = torch.from_numpy(self.x).type(torch.float32)
        self.y = torch.from_numpy(self.y).type(torch.float32)

        train_split = 0.8
        train_idx = int(len(self.cam_paths) * train_split)

        if train:
            self.cam_paths = self.cam_paths[:train_idx]
            self.meta = self.meta[:train_idx]
        else:
            self.cam_paths = self.cam_paths[train_idx:]
            self.meta = self.meta[:train_idx:]

        print("Total number of images: %i" % len(self.cam_paths))

    def _pil_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img = transforms.functional.pil_to_tensor(img)
            target_size = (self.target_size, self.target_size)
            img = transforms.functional.resize(img, target_size)
            img = img.type(torch.float32) / 255
            return img

    def __len__(self):
        return len(self.map_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        cam_img = self._pil_loader(self.cam_paths[idx])
        meta = self.meta[idx]
        if self.transform is not None:
            cam_img = self.transform(cam_img)
        return cam_img, meta

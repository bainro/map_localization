import argparse
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Config
train_split = 0.8

class PerspectiveTransformTorchDataset(Dataset):
    def __init__(self, source, train=True, transform=None, target_size=128, **kwargs):

        super().__init__(**kwargs)
        self.camera_paths = []
        self.meta = []
        self.transform = transform
        self.target_size = target_size

        for dir in os.listdir(source):
            if os.path.isdir(os.path.join(source, dir)) and not dir.startswith('.'):
                source_dir = os.path.join(source, dir, "extracted_frames")
                try:
                    all_meta = pd.read_csv(os.path.join(source_dir, "meta_data.csv"), index_col=0)
                    all_meta = all_meta[['time', 'heading', 'x', 'y']]
                except pd.errors.EmptyDataError:
                    continue
                files = os.listdir(source_dir)
                cam_pic_files = [f for f in files if '_camera.png' in f]

                idxs = []
                for cam_f in cam_pic_files:
                    idx = cam_f.replace('_camera.png', '').split('_')[-1]
                    idxs.append(int(idx))
                idxs = sorted(idxs)

                for idx in idxs:
                    self.camera_paths.append(os.path.join(source_dir, "%d_camera.png" % idx))
                    meta = all_meta.loc[idx].to_numpy()
                    self.meta.append(meta)

        self.camera_paths = np.array(self.camera_paths)
        self.meta = torch.from_numpy(np.stack(self.meta)).type(torch.float32)

        if self.debug and len(self.camera_paths) > self.debug_size:
            p = np.random.permutation(len(self.camera_paths))
            self.camera_paths = self.camera_paths[p][:self.debug_size]
            self.meta = self.meta[p][:self.debug_size]

        train_idx = int(len(self.camera_paths) * train_split)

        if train:
            self.camera_paths = self.camera_paths[:train_idx]
            self.meta = self.meta[:train_idx]
        else:
            self.camera_paths = self.camera_paths[train_idx:]
            self.meta = self.meta[:train_idx:]

        print("Total number of images: %i" % len(self.map_paths))

    def _pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
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
        camera = self._pil_loader(self.camera_paths[idx])
        meta = self.meta[idx]
        if self.transform is not None:
            camera = self.transform(camera)

        return camera, meta

    def get_label_dim(self):
        return self.meta.shape[-1]

    def get_shape(self):
        source, target, meta = self[0]
        return source.shape[1:], target.shape[1:]

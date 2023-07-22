import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class LocalizeDataset(Dataset):
    def __init__(self, src_dir, train=True, transform=None, 
                 shuffle=True, target_size=128, **kwargs):

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
        assert len(cam_pic_files) > 0, "no camera images found :("

        idxs = []
        for cam_f in cam_pic_files:
            idx = cam_f.replace('_camera.png', '').split('_')[-1]
            idxs.append(int(idx))
        idxs = sorted(idxs)
        if shuffle:
            random.Random(42).shuffle(idxs)

        for idx in idxs:
            cam_p = os.path.join(src_dir, "%d_camera.png" % idx)
            self.cam_paths.append(cam_p)
            x = meta_x.loc[idx].to_numpy()
            y = meta_y.loc[idx].to_numpy()
            self.x.append(x)
            self.y.append(y)

        self.cam_paths = np.array(self.cam_paths)
        self.x = torch.FloatTensor(np.array(self.x))
        self.y = torch.FloatTensor(np.array(self.y))

        train_split = 0.85
        train_idx = int(len(self.cam_paths) * train_split)

        if train:
            self.cam_paths = self.cam_paths[:train_idx]
            self.x = self.x[:train_idx]
            self.y = self.y[:train_idx]
        else:
            self.cam_paths = self.cam_paths[train_idx:]
            self.x = self.x[train_idx:]
            self.y = self.y[train_idx:]

        print("Total number of images: %i" % len(self.cam_paths))

    def _pil_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img = transforms.functional.pil_to_tensor(img)
            target_size = (self.target_size, self.target_size)
            img = transforms.functional.resize(img, target_size, antialias=True)
            img = img.type(torch.float32) / 255
            return img

    def __len__(self):
        return len(self.cam_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        cam_img = self._pil_loader(self.cam_paths[idx])
        x = self.x[idx]
        y = self.y[idx]
        if self.transform is not None:
            cam_img = self.transform(cam_img)
        return cam_img, x, y

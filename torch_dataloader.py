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
att_width = 8


class PerspectiveTransformTorchDataset(Dataset):
    def __init__(self, source, train=True, transform=None, target_size=128, square=True, map2camera=True, debug=False,
                 debug_size=100, **kwargs):

        super().__init__(**kwargs)
        self.map_paths = []
        self.camera_paths = []
        self.meta = []
        self.transform = transform
        self.target_size = target_size
        self.square = square
        self.map2camera = map2camera
        self.debug = debug
        self.debug_size = debug_size

        for dir in os.listdir(source):
            if os.path.isdir(os.path.join(source, dir)) and not dir.startswith('.'):
                source_dir = os.path.join(source, dir, "extracted_frames")
                try:
                    all_meta = pd.read_csv(os.path.join(source_dir, "meta_data.csv"), index_col=0)
                    all_meta = all_meta[['time', 'heading']]
                except pd.errors.EmptyDataError:
                    continue
                files = os.listdir(source_dir)
                map_pic_files = [f for f in files if '_map.png' in f]

                idxs = []
                for map_f in map_pic_files:
                    idx = map_f.replace('_map.png', '').split('_')[-1]
                    idxs.append(int(idx))
                idxs = sorted(idxs)

                for idx in idxs:
                    self.map_paths.append(os.path.join(source_dir, "%d_map.png" % idx))
                    self.camera_paths.append(os.path.join(source_dir, "%d_camera.png" % idx))
                    meta = all_meta.loc[idx].to_numpy()
                    self.meta.append(meta)

        self.map_paths = np.array(self.map_paths)
        self.camera_paths = np.array(self.camera_paths)
        self.meta = torch.from_numpy(np.stack(self.meta)).type(torch.float32)

        if self.debug and len(self.map_paths) > self.debug_size:
            p = np.random.permutation(len(self.map_paths))
            self.map_paths = self.map_paths[p][:self.debug_size]
            self.camera_paths = self.camera_paths[p][:self.debug_size]
            self.meta = self.meta[p][:self.debug_size]

        train_idx = int(len(self.map_paths) * train_split)

        if train:
            self.map_paths = self.map_paths[:train_idx]
            self.camera_paths = self.camera_paths[:train_idx]
            self.meta = self.meta[:train_idx]
        else:
            self.map_paths = self.map_paths[train_idx:]
            self.camera_paths = self.camera_paths[train_idx:]
            self.meta = self.meta[:train_idx:]

        print("Total number of images: %i" % len(self.map_paths))

    def _pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img = transforms.functional.pil_to_tensor(img)

            c, h, w = img.shape
            # if the image's aspect ratio is too wide, use a 2:1 aspect ratio
            # Crop the center rectangle based on aspect ratio of size
            target_size = (self.target_size, self.target_size)
            if w/h >= 2:
                if self.square:
                    # Use padding instead of cropping
                    new_h = max(h, w)
                    new_w = max(h, w)
                    padded_img = torch.zeros([c, new_h, new_w])
                    padded_img[:, (new_h - h) // 2:(new_h + h) // 2, (new_w - w) // 2:(new_w + w) // 2] = img
                    img = padded_img
                else:
                    target_size = (self.target_size,  self.target_size * 2)
            else:
                # Crop the center square
                new_h = min(h, w)
                new_w = min(h, w)
                img = img[:, (h - new_h) // 2:(h + new_h) // 2, (w - new_w) // 2:(w + new_w) // 2]

            img = transforms.functional.resize(img, target_size)
            img = img.type(torch.float32) / 255
            return img

    def __len__(self):
        return len(self.map_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        map = self._pil_loader(self.map_paths[idx])
        camera = self._pil_loader(self.camera_paths[idx])
        meta = self.meta[idx]
        if self.transform is not None:
            map = self.transform(map)
            camera = self.transform(camera)

        if self.map2camera:
            source, target = map, camera
        else:
            source, target = camera, map

        return source, target, meta

    def get_label_dim(self):
        return self.meta.shape[-1]

    def get_shape(self):
        source, target, meta = self[0]
        return source.shape[1:], target.shape[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="data")
    args = parser.parse_args()

    train_dataset = PerspectiveTransformTorchDataset(args.source, square=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    map_shape, camera_shape = train_dataset.get_shape()
    print("Map shape: %r, Camera shape: %r" % (map_shape, camera_shape))

    for map, camera, meta in train_loader:
        idx = np.random.randint(len(map))
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        axs[0].imshow(camera[idx].movedim(0, -1))
        axs[1].imshow(map[idx].movedim(0, -1))
        plt.savefig('inspect_img.png')

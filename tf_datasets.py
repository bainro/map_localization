import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

AUTO = tf.data.AUTOTUNE
DEBUG_SAMPLES = 50


class PerspectiveTransformTFDataset:
    def __init__(self, source, map2camera=True, batch_size=4, img_size=128, debug=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.source = source

        self.map_paths = []
        self.camera_paths = []
        self.meta = []
        self.map2camera = map2camera
        self.debug = debug

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

                    if self.debug and len(self.map_paths) >= DEBUG_SAMPLES:
                        break

                if self.debug and len(self.map_paths) >= DEBUG_SAMPLES:
                    break

        self.meta = np.stack(self.meta)
        self.map_paths = np.array(self.map_paths)
        self.camera_paths = np.array(self.camera_paths)
        self.rescaler = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)
        print("Total number of images: %i" % len(self.map_paths))

    def process_image(self, map_path, camera_path):
        map = tf.io.read_file(map_path)
        map = tf.io.decode_png(map, 3)
        map = tf.keras.preprocessing.image.smart_resize(map, (self.img_size, self.img_size))
        map = self.rescaler(map)

        camera = tf.io.read_file(camera_path)
        camera = tf.io.decode_png(camera, 3)
        camera = tf.keras.preprocessing.image.smart_resize(camera, (self.img_size, self.img_size))
        camera = self.rescaler(camera)

        if self.map2camera:
            return map, camera
        else:
            return camera, map

    def prepare_dataset(self, split="train"):
        train_end = int(len(self.map_paths) * 0.7)
        valid_end = int(len(self.map_paths) * 0.85)
        test_end = len(self.map_paths)

        if split == "train":
            start_ix = 0
            end_ix = train_end
        elif split == "valid":
            start_ix = train_end
            end_ix = valid_end
        elif split == "test":
            start_ix = valid_end
            end_ix = test_end
        map_paths = self.map_paths[start_ix:end_ix]
        camera_paths = self.camera_paths[start_ix:end_ix]
        dataset = tf.data.Dataset.from_tensor_slices((map_paths, camera_paths))
        dataset = dataset.shuffle(self.batch_size * 10)
        dataset = dataset.map(self.process_image, num_parallel_calls=AUTO).batch(
            self.batch_size, drop_remainder=True
        )
        return dataset.prefetch(AUTO)


if __name__ == "__main__":
    data_utils = PerspectiveTransformTFDataset("data")
    training_dataset = data_utils.prepare_dataset()

    for map, camera in training_dataset:
        idx = np.random.randint(len(map))
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(map[idx])
        axs[1].imshow(camera[idx])
        plt.savefig('inspect_img.png')
        break

"""
Adapted from  https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

# Usage
python diffuser_finetune.py
"""

import warnings

warnings.filterwarnings("ignore")

import os
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from datetime import datetime

import tensorflow as tf
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from tensorflow.keras import mixed_precision
import numpy as np

from tf_datasets import PerspectiveTransformTFDataset
from diffuser_trainer import Trainer

MAX_PROMPT_LENGTH = 77


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to fine-tune a Stable Diffusion model."
    )
    # Dataset related.
    parser.add_argument("--dataset_archive", default=None, type=str)
    parser.add_argument("--img_size", default=128, type=int)
    # Optimization hyperparameters.
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--wd", default=1e-2, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--epsilon", default=1e-08, type=float)
    parser.add_argument("--ema", default=0.9999, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    # Training hyperparameters.
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument('--map2camera', default=False, action='store_true')
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--debug', default=False, action="store_true")

    # Others.
    parser.add_argument(
        "--mp", action="store_true", help="Whether to use mixed-precision."
    )
    parser.add_argument(
        "--pretrained_ckpt",
        default=None,
        type=str,
        help="Provide a local path to a diffusion mocheckpoint in the `h5`"
        " format if you want to start over fine-tuning from this checkpoint.",
    )

    return parser.parse_args()


def run(args):

    now = datetime.now()
    if args.debug:
        root_dir = "debug_runs"
    else:
        root_dir = "runs"
    log_dir = os.path.join(root_dir, "%s_%s" %(args.tag, now.strftime("%m_%d_%Y_%H_%M_%S")))

    if args.mp:
        print("Enabling mixed-precision...")
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        assert policy.compute_dtype == "float16"
        assert policy.variable_dtype == "float32"

    print("Initializing dataset...")
    data_utils = PerspectiveTransformTFDataset("data", img_size=args.img_size, batch_size=args.batch_size, map2camera=args.map2camera, debug=args.debug)
    train_dataset = data_utils.prepare_dataset(split="train")
    valid_dataset = data_utils.prepare_dataset(split="valid")
    test_dataset = data_utils.prepare_dataset(split="test")

    print("Initializing trainer...")
    image_encoder = ImageEncoder(args.img_size, args.img_size)
    trainer = Trainer(
        diffusion_model=DiffusionModel(
            args.img_size, args.img_size, MAX_PROMPT_LENGTH
        ),
        # Remove the top layer from the encoder, which cuts off the variance and only returns
        # the mean
        vae=tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-2].output,
        ),
        noise_scheduler=NoiseScheduler(),
        pretrained_ckpt=args.pretrained_ckpt,
        mp=args.mp,
        ema=args.ema,
        max_grad_norm=args.max_grad_norm,
        args=args
    )

    print("Initializing optimizer...")
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=args.lr,
        weight_decay=args.wd,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        epsilon=args.epsilon,
    )
    if args.mp:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    print("Compiling trainer...")
    trainer.compile(optimizer=optimizer, loss="mse")
    # trainer.compile(optimizer=optimizer, loss="mse", run_eagerly=True)


    class TensorBoardImage(tf.keras.callbacks.TensorBoard):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def normalize_img(self, img):
            img = ((img + 1) / 2) * 255
            return np.clip(img, 0, 255).astype("uint8")

        def generate_image_to_tensorboard(self, batch, writer):
            source, target = batch
            predict = trainer.generate_image(batch)
            source = self.normalize_img(source)
            target = self.normalize_img(target)
            predict = self.normalize_img(predict)

            with tf.summary.record_if(True):
                with writer.as_default():
                    tf.summary.image("source", source, step=args.num_epochs)
                    tf.summary.image("target", target, step=args.num_epochs)
                    tf.summary.image("predict", predict, step=args.num_epochs)

        def on_epoch_end(self, epoch, logs=None):
            self.generate_image_to_tensorboard(iter(train_dataset).get_next(), self._train_writer)
            self.generate_image_to_tensorboard(iter(valid_dataset).get_next(), self._val_writer)
            super().on_epoch_end(epoch, logs)
            return


    print("Training...")
    tensorboard_callback = TensorBoardImage(log_dir,  write_graph=False)
    tensorboard_callback.set_model(trainer)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(log_dir, monitor="val_diffuse_loss", mode="min", save_weights_only=True, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(),
        tensorboard_callback
    ]
    trainer.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=args.num_epochs,
        callbacks=callbacks
    )

    test_writer = tf.summary.create_file_writer(os.path.join(tensorboard_callback._get_log_write_dir(), "test"))
    tensorboard_callback.generate_image_to_tensorboard(iter(test_dataset).get_next(), test_writer)

    test_results = trainer.evaluate(test_dataset)
    print("Test reconstruction loss: %r" % test_results)
    with tf.summary.record_if(True):
        with test_writer.as_default():
            tf.summary.scalar("loss", test_results[0], step=args.num_epochs)
            tf.summary.scalar("diffuse_loss", test_results[1], step=args.num_epochs)
            tf.summary.scalar("recon_loss", test_results[2], step=args.num_epochs)
            tf.summary.scalar("kl_loss", test_results[3], step=args.num_epochs)

if __name__ == "__main__":
    args = parse_args()
    run(args)

# python diffuser_finetune.py --batch_size 4 --num_epochs 1 --tag debug --debug --map2camera

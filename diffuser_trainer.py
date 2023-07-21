import math

import numpy as np

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.constants import _ALPHAS_CUMPROD
from keras_cv.models.stable_diffusion.constants import _UNCONDITIONAL_TOKENS
from keras_cv.models.stable_diffusion.decoder import Decoder
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder

MAX_PROMPT_LENGTH = 77

class Trainer(tf.keras.Model):
    # Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

    def __init__(
        self,
        diffusion_model: tf.keras.Model,
        vae: tf.keras.Model,
        noise_scheduler: NoiseScheduler,
        pretrained_ckpt: str,
        mp: bool,
        args,
        ema=0.9999,
        max_grad_norm=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diffusion_model = diffusion_model
        if pretrained_ckpt is not None:
            self.diffusion_model.load_weights(pretrained_ckpt)
            print(
                f"Loading the provided checkpoint to initialize the diffusion model: {pretrained_ckpt}..."
            )

        self.vae = vae
        self.noise_scheduler = noise_scheduler

        if ema > 0.0:
            self.ema = tf.Variable(ema, dtype="float32")
            self.optimization_step = tf.Variable(0, dtype="int32")
            self.ema_diffusion_model = DiffusionModel(
                args.img_size, args.img_size, 77
            )
            self.do_ema = True
        else:
            self.do_ema = False

        self.img_size = args.img_size
        self.vae.trainable = True
        self.mp = mp
        self.max_grad_norm = max_grad_norm
        self.fit_text_embedding = tf.keras.layers.Dense(77*768)
        self.decoder = Decoder(args.img_size, args.img_size)
        self.text_encoder = TextEncoder(MAX_PROMPT_LENGTH)

    @staticmethod
    def KL(mu, logsigma):
        loss = -0.5 * (1 + logsigma - tf.square(mu) - tf.exp(logsigma))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        return loss

    def forward(self, inputs):
        source_images, target_images = inputs
        context, ctx_mu, ctx_logsigma = self.sample_from_encoder_outputs(self.vae(source_images, training=False))
        ctx_kl_loss = self.KL(ctx_mu, ctx_logsigma)
        bsz = tf.shape(context)[0]

        context = self.fit_text_embedding(tf.reshape(context, [-1, np.prod(self.vae.output.shape[1:]) // 2]))
        context = tf.reshape(context, [-1, 77, 768])

        # Project image into the latent space.
        latents, lat_mu, lat_logsigma = self.sample_from_encoder_outputs(self.vae(target_images, training=False))
        lat_kl_loss = self.KL(lat_mu, lat_logsigma)
        kl_loss = ctx_kl_loss + lat_kl_loss
        kl_loss /= self.img_size**2
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = tf.random.normal(tf.shape(latents))

        # Sample a random timestep for each image
        timesteps = tnp.random.randint(
            0, self.noise_scheduler.train_timesteps, (bsz,)
        )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(
            tf.cast(latents, noise.dtype), noise, timesteps
        )

        # Predict the noise residual and compute loss
        timestep_embeddings = tf.map_fn(
            lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
        )
        timestep_embeddings = tf.squeeze(timestep_embeddings, 1)
        model_pred = self.diffusion_model(
            [noisy_latents, timestep_embeddings, context], training=True
        )

        diffuse_loss = self.compiled_loss(noise, model_pred)

        # Also need to tune the decoder
        recon = self.decoder(tf.stop_gradient(latents))
        recon_loss = self.compiled_loss(recon, target_images)

        loss = diffuse_loss + recon_loss + kl_loss
        # loss = diffuse_loss
        if self.mp:
            loss = self.optimizer.get_scaled_loss(loss)

        logs = {'loss': loss,
                'diffuse_loss': diffuse_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss}

        return model_pred, loss, logs

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            model_pred, loss, logs = self.forward(inputs)

        # Update parameters of the diffusion model and vae
        # trainable_vars = self.diffusion_model.trainable_variables + self.vae.trainable_variables + self.decoder.trainable_variables
        trainable_vars = self.diffusion_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        if self.mp:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        if self.max_grad_norm > 0.0:
            gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # EMA.
        if self.do_ema:
            self.ema_step()

        logs = {m.name: m.result() for m in self.metrics}
        return logs

    def test_step(self, inputs):
        model_pred, loss, logs = self.forward(inputs)
        return logs

    def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
        # Taken from
        # # https://github.com/keras-team/keras-cv/blob/ecfafd9ea7fe9771465903f5c1a03ceb17e333f1/keras_cv/models/stable_diffusion/stable_diffusion.py#L481
        half = dim // 2
        log_max_period = tf.math.log(tf.cast(max_period, tf.float32))
        freqs = tf.math.exp(-log_max_period * tf.range(0, half, dtype=tf.float32) / half)
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return embedding  # Excluding the repeat.

    def get_decay(self, optimization_step):
        value = (1 + optimization_step) / (10 + optimization_step)
        value = tf.cast(value, dtype=self.ema.dtype)
        return 1 - tf.math.minimum(self.ema, value)

    def ema_step(self):
        self.optimization_step.assign_add(1)
        self.ema.assign(self.get_decay(self.optimization_step))

        for weight, ema_weight in zip(
            self.diffusion_model.trainable_variables,
            self.ema_diffusion_model.trainable_variables,
        ):
            tmp = self.ema * (ema_weight - weight)
            ema_weight.assign_sub(tmp)

    def sample_from_encoder_outputs(self, outputs):
        mean, logsigma = tf.split(outputs, 2, axis=-1)
        logsigma = tf.clip_by_value(logsigma, -30.0, 20.0)
        std = tf.exp(0.5 * logsigma)
        sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + std * sample, mean, logsigma

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Overriding to help with the `ModelCheckpoint` callback.
        if self.do_ema:
            self.ema_diffusion_model.save_weights(
                filepath=filepath+"/diffusion.h5",
                overwrite=overwrite,
                save_format=save_format,
                options=options,
            )
        else:
            self.diffusion_model.save_weights(
                filepath=filepath+"/diffusion.h5",
                overwrite=overwrite,
                save_format=save_format,
                options=options,
            )
        self.vae.save_weights(
            filepath=filepath+"/vae.h5",
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        self.decoder.save_weights(
            filepath=filepath+"/decoder.h5",
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # Overriding to help with the `ModelCheckpoint` callback.
        if self.do_ema:
            self.ema_diffusion_model.load_weights(filepath + "/diffusion.h5")
        else:
            self.diffusion_model.load_weights(filepath=filepath)
        self.vae.load_weights(filepath+"/vae.h5")
        self.decoder.load_weights(filepath+"/decoder.h5")

    @staticmethod
    def _get_pos_ids():
        return tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)

    def _get_unconditional_context(self):
        unconditional_tokens = tf.convert_to_tensor(
            [_UNCONDITIONAL_TOKENS], dtype=tf.int32
        )
        unconditional_context = self.text_encoder.predict_on_batch(
            [unconditional_tokens, self._get_pos_ids()]
        )

        return unconditional_context

    def _get_initial_alphas(self, timesteps):
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev

    def _get_initial_diffusion_noise(self, batch_size, seed):
        if seed is not None:
            return tf.random.stateless_normal(
                (batch_size, self.img_size // 8, self.img_size // 8, 4),
                seed=[seed, seed],
            )
        else:
            return tf.random.normal(
                (batch_size, self.img_size // 8, self.img_size // 8, 4)
            )

    def _batch_get_timestep_embedding(self, timestep, batch_size, dim=320, max_period=10000):
        half = dim // 2
        freqs = tf.math.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

    def generate_image(
        self,
        inputs,
        num_steps=50,
        unconditional_guidance_scale=7.5,
    ):

        source_images, target_images = inputs
        context, _, _ = self.sample_from_encoder_outputs(self.vae(source_images, training=False))
        context = self.fit_text_embedding(tf.reshape(context, [-1, np.prod(self.vae.output.shape[1:]) // 2]))
        context = tf.reshape(context, [-1, 77, 768])
        batch_size = tf.shape(context)[0]
        unconditional_context = tf.repeat(
            self._get_unconditional_context(), batch_size, axis=0
        )
        latent = self._get_initial_diffusion_noise(batch_size, None)

        # Iterative reverse diffusion stage
        timesteps = tf.range(1, 1000, 1000 // num_steps)
        alphas, alphas_prev = self._get_initial_alphas(timesteps)
        # progbar = tf.keras.utils.Progbar(len(timesteps))
        # iteration = 0
        for index, timestep in list(enumerate(timesteps))[::-1]:
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = self._batch_get_timestep_embedding(timestep, batch_size)
            unconditional_latent = self.diffusion_model(
                [latent, t_emb, unconditional_context]
            )
            latent = self.diffusion_model([latent, t_emb, context])
            latent = unconditional_latent + unconditional_guidance_scale * (
                latent - unconditional_latent
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(a_t)
            latent = latent * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0
            # iteration += 1
            # progbar.update(iteration)

        # Decoding stage
        predict_images = self.decoder.predict_on_batch(latent)
        return predict_images

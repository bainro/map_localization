from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import densenet121
from torchvision.transforms._presets import ImageClassification

from vae_model import GeneralEncoder, GeneralDecoder4


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Encoder(nn.Module):
    def __init__(self, img_size, latent_size=128):
        super(Encoder, self).__init__()
        self.img_size = img_size
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2, stride=2)  # in_channels=3
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)
        self.fc1 = nn.Linear(256 * (self.img_size//8) ** 2, 2048)
        self.bn4 = nn.BatchNorm1d(2048, momentum=0.9)
        self.fc_mean = nn.Linear(2048, latent_size)
        self.fc_logsigma = nn.Linear(2048, latent_size)  # latent dim=128
        self.apply(weights_init)

    def forward(self, x):
        batch_size = x.size()[0]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = out.view(batch_size, -1)
        out = self.relu(self.bn4(self.fc1(out)))
        mean = self.fc_mean(out)
        logsigma = self.fc_logsigma(out)
        return mean, logsigma


class Decoder(nn.Module):
    def __init__(self, img_size, latent_size=128):
        super(Decoder, self).__init__()
        self.img_size = img_size
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 256 * (self.img_size//8) ** 2)
        self.bn1 = nn.BatchNorm1d(256 * (self.img_size//8) ** 2, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)
        self.deconv1 = nn.ConvTranspose2d(256, 256, 6, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 6, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.deconv3 = nn.ConvTranspose2d(128, 32, 6, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2)
        self.tanh = nn.Tanh()
        self.apply(weights_init)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 256, 8, 8)
        x = self.relu(self.bn2(self.deconv1(x)))
        x = self.relu(self.bn3(self.deconv2(x)))
        x = self.relu(self.bn4(self.deconv3(x)))
        x = self.tanh(self.deconv4(x))
        return x


class Decoder_2(nn.Module):
    def __init__(self, latent_size=256, ngf=64, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_size, ngf * 12, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 12),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ngf*8) x 5 x 5
            nn.ConvTranspose2d(ngf * 12, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ngf*4) x 10 x 10
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ngf*2) x 20 x 20
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ngf) x 40 x 40
            nn.ConvTranspose2d(ngf * 2,  ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ngf) x 80 x 80
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 160 x 160
        )

    def forward(self, input):
        return self.main(input.view(input.shape[0], -1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, img_size):
        self.img_size = img_size

        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2, stride=1)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(32, 128, 5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.conv4 = nn.Conv2d(256, 256, 5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9)
        self.fc1 = nn.Linear(256 * (self.img_size//8) ** 2, 512)
        self.bn4 = nn.BatchNorm1d(512, momentum=0.9)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.bn3(self.conv4(x)))
        x = x.view(-1, 256 * (self.img_size//8) ** 2)
        features = x;
        x = self.relu(self.bn4(self.fc1(features)))
        out = self.sigmoid(self.fc2(x))

        return out, features


class VAEGAN1(nn.Module):
    def __init__(self, img_size, latent_size=128):
        super().__init__()
        self.img_size = img_size
        self.latent_size = latent_size

        # Encoder
        self.encoder = Encoder(img_size, latent_size)
        self.generator = Decoder(img_size, latent_size)
        self.discriminator = Discriminator(img_size)

    def reparameterize(self, mu, logsigma):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = self.reparameterize(mu, logsigma)
        return self.generator(z), mu, logsigma


class DensenetEncoder(nn.Module):
    def __init__(self, transform, latent_size=128):
        super().__init__()
        self.densenet = densenet121(weights='DEFAULT')
        self.transform = transform
        self.latent_size = latent_size
        num_ftrs = self.densenet.classifier.in_features
        self.fc_mean = nn.Linear(num_ftrs, latent_size)
        self.fc_logsigma = nn.Linear(num_ftrs, latent_size)  # latent dim=128


    def forward(self, x):
        features = self.densenet.features(self.transform(x))
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        mean = self.fc_mean(features)
        logsigma = self.fc_logsigma(features)
        return mean, logsigma


class DensenetDiscriminator(nn.Module):
    def __init__(self, transform, latent_size=128):
        super().__init__()
        self.densenet = densenet121(weights='DEFAULT')
        self.transform = transform
        self.latent_size = latent_size
        num_ftrs = self.densenet.classifier.in_features
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.bn4 = nn.BatchNorm1d(512, momentum=0.9)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        features = self.densenet.features(self.transform(x))
        features = F.relu(features, inplace=True)
        features_ = features
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        x = F.relu(self.bn4(self.fc1(features)))
        out = self.sigmoid(self.fc2(x))
        return out, features_


class VAEGAN_DENSENET(nn.Module):
    def __init__(self, img_size, latent_size=128):
        super().__init__()

        # Encoder
        self.transform = partial(ImageClassification, crop_size=img_size, resize_size=img_size)()
        self.encoder = DensenetEncoder(self.transform, latent_size)
        self.generator = Decoder(img_size, latent_size)
        self.discriminator = DensenetDiscriminator(self.transform, latent_size)


    def reparameterize(self, mu, logsigma):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = self.reparameterize(mu, logsigma)
        return self.generator(z), mu, logsigma


class VAEGAN2(nn.Module):
    def __init__(self, img_size, latent_size=128):
        super().__init__()

        self.encoder = Encoder(img_size, latent_size)
        self.generator = Decoder_2(latent_size)
        self.discriminator = Discriminator(img_size)

    def reparameterize(self, mu, logsigma):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = self.reparameterize(mu, logsigma)
        return self.generator(z), mu, logsigma


class VAEGAN_JINWEI(nn.Module):
    def __init__(self, img_size, latent_size=128):
        super().__init__()

        self.encoder = GeneralEncoder((img_size, img_size), latent_size=latent_size, attn=False)
        self.generator = GeneralDecoder4((img_size, img_size), latent_size=latent_size)
        self.discriminator = Discriminator(img_size)
        self.latent_size = latent_size

    def reparameterize(self, mu, logsigma):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, meta):
        mu, logsigma, _, _, _ = self.encoder(x, meta)
        z = self.reparameterize(mu, logsigma)
        return self.generator(z), mu, logsigma


# The followings are taken here
# https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/nn/benchmarks/cifar/convnets.py
# and modified to fit our needs

class Encoder_Conv_VAE_CIFAR(nn.Module):

    def __init__(self, img_size, latent_size=128):
        super().__init__(self)

        self.input_dim = (3, img_size, img_size)
        self.latent_dim = latent_size
        self.n_channels = 3
        layers = nn.ModuleList()
        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )
        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1), nn.BatchNorm2d(256), nn.ReLU()
            )
        )
        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1), nn.BatchNorm2d(512), nn.ReLU()
            )
        )
        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1), nn.BatchNorm2d(1024), nn.ReLU()
            )
        )
        self.layers = layers
        self.depth = len(layers)
        self.linear_mu = nn.Linear(1024 * 8 * 8, latent_size)
        self.linear_logsigma = nn.Linear(1024 * 8 * 8, latent_size)

    def forward(self, x):
        for i in range(self.depth):
            x = self.layers[i](x)
        mu = self.linear_mu(x.reshape(x.shape[0], -1))
        logsigma = self.linear_logsigma(x.reshape(x.shape[0], -1))
        return mu, logsigma


class Decoder_Conv_AE_CIFAR(nn.Module):

    def __init__(self, img_size, latent_size=128):
        super().__init__(self)
        self.input_dim = (3, img_size, img_size)
        self.latent_size = latent_size
        self.n_channels = 3
        layers = nn.ModuleList()
        layers.append(nn.Linear((latent_size), 1024 * 8 * 8))
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 4, 2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, self.n_channels, 4, 1, padding=2), nn.Sigmoid()
            )
        )
        self.layers = layers
        self.depth = len(layers)

    def forward(self, z):
        out = z
        for i in range(self.depth):
            out = self.layers[i](out)
            if i == 0:
                out = out.reshape(z.shape[0], 1024, 8, 8)
        return out


class Discriminator_Conv_CIFAR(nn.Module):

    def __init__(self, img_size, latent_size=128):
        super().__init__(self)
        self.input_dim = (3, img_size, img_size)
        self.latent_size = latent_size
        self.n_channels = 3
        layers = nn.ModuleList()
        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.ReLU(),
            )
        )
        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1),
                nn.Tanh(),
            )
        )
        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1),
                nn.ReLU(),
            )
        )
        layers.append(nn.Sequential(nn.Linear(1024 * 2 * 2, 1), nn.Sigmoid()))
        self.layers = layers
        self.depth = len(layers)

    def forward(self, x):
        max_depth = self.depth
        out = x
        for i in range(max_depth):
            if i == 4:
                out = out.reshape(x.shape[0], -1)
                features = out
            out = self.layers[i](out)
        return out, features

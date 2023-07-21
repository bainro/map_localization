import torch
import torch.nn as nn


def reparameterize(mu, logsigma):
    std = torch.exp(0.5 * logsigma)
    eps = torch.randn_like(std)
    return mu + eps * std


class Attn(nn.Module):
    """Attention Layer

    code obtained from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """

    def __init__(self, query_dim, key_dim, activation):
        super().__init__()
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=query_dim, out_channels=key_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=key_dim, out_channels=key_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=key_dim, out_channels=query_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, query, key, value):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, q_channel, q_width, q_height = query.size()
        m_batchsize, k_channel, k_width, k_height = key.size()
        m_batchsize, v_channel, v_width, v_height = value.size()
        proj_query = self.query_conv(query).view(m_batchsize, -1, q_width * q_height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(key).view(m_batchsize, -1, k_width * k_height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (*W*H)
        proj_value = self.value_conv(value).view(m_batchsize, -1, v_width * v_height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, q_channel, q_width, q_height)

        out = self.gamma * out + query
        return out, attention


class GeneralEncoder(nn.Module):
    def __init__(self, img_shape, latent_size=10, attn=False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 2, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=2), nn.ReLU()
        )
        self.feature_attn = Attn(32, 32, 'relu')
        self.mu_attn = Attn(1, 32, 'relu')
        self.logsigma_attn = Attn(1, 32, 'relu')
        self.img_shape = img_shape
        # Config for 360 camera
        flatten_size = int(32 * (img_shape[0] / 16) * (img_shape[1] / 16))
        flatten_size += 2  # This is for the meta data
        self.linear_mu = nn.Linear(flatten_size, latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, latent_size)
        self.attn = attn

    def forward(self, x, meta):
        features = self.main(x)
        # features, features_attn_map = self.feature_attn(features, features, features)
        x = torch.flatten(features, start_dim=1)
        x = torch.cat([x, meta], axis=1)
        mu = self.linear_mu(x)
        mu = torch.unsqueeze(torch.unsqueeze(mu, 1), -1)  # put mu dim as spatial dim
        if self.attn:
            mu, mu_attn_map = self.mu_attn(mu, features, features)
        else:
            mu_attn_map = torch.zeros_like(mu)
        mu = torch.squeeze(mu)
        logsigma = self.linear_logsigma(x)
        logsigma = torch.unsqueeze(torch.unsqueeze(logsigma, 1), -1)
        if self.attn:
            logsigma, logsigma_attn_map = self.logsigma_attn(logsigma, features, features)
        else:
            logsigma_attn_map = torch.zeros_like(logsigma)
        logsigma = torch.squeeze(logsigma)
        return mu, logsigma, features, mu_attn_map, logsigma_attn_map


class MapEncoder(nn.Module):
    def __init__(self, img_shape, latent_size=10, attn=False):
        """This specifically take the meta-data as input and cocnatenate it to the latent space"""
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 2, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=2), nn.ReLU()
        )
        self.img_shape = img_shape
        flatten_size = int(32 * (img_shape[0] / 16) * (img_shape[1] / 16))
        flatten_size += 2  # This is for the meta data
        self.linear_mu = nn.Linear(flatten_size, latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, latent_size)
        self.attn = attn

    def forward(self, x, meta):
        features = self.main(x)
        x = torch.flatten(features, start_dim=1)
        x = torch.cat([x, meta], axis=1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        return mu, logsigma, features


class GeneralDecoder(nn.Module):
    def __init__(self, img_shape, latent_size=10):
        super().__init__()

        # 360 camera config
        self.img_shape = img_shape
        self.aspect_ratio = int(img_shape[1] / img_shape[0])
        self.flatten_size = 1024
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.flatten_size, 128, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=4), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 2, stride=2),
            nn.Sigmoid()
        )
        # We want to use half of the latent size as the extra length of the aspect ratio
        self.fc = nn.Linear(latent_size, int(self.flatten_size * self.aspect_ratio))

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.reshape(-1, self.flatten_size, 1, self.aspect_ratio)
        out = self.main(x)
        return out


class GeneralDecoder2(nn.Module):
    def __init__(self, img_shape, latent_size=10):
        super().__init__()

        # 360 camera config
        self.img_shape = img_shape
        self.aspect_ratio = int(img_shape[1] / img_shape[0])
        self.flatten_size = 1024
        # this output shape 128x128
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.flatten_size, 512, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),  # Add this for 256
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        # We want to use half of the latent size as the extra length of the aspect ratio
        self.fc = nn.Linear(latent_size, int(self.flatten_size * self.aspect_ratio))

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.reshape(-1, self.flatten_size, 1, self.aspect_ratio)
        out = self.main(x)
        return out


class GeneralDecoder3(nn.Module):
    def __init__(self, img_shape, latent_size=10):
        super().__init__()

        # 360 camera config
        self.img_shape = img_shape
        self.aspect_ratio = int(img_shape[1] / img_shape[0])
        self.flatten_size = 1024
        # this output shape 128x128
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.flatten_size, 512, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # Add this for 256
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        # We want to use half of the latent size as the extra length of the aspect ratio
        self.fc = nn.Linear(latent_size, int(self.flatten_size * self.aspect_ratio))

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.reshape(-1, self.flatten_size, 1, self.aspect_ratio)
        out = self.main(x)
        return out


class GeneralDecoder4(nn.Module):
    def __init__(self, img_shape, latent_size=10):
        super().__init__()

        # 360 camera config
        self.img_shape = img_shape
        self.aspect_ratio = int(img_shape[1] / img_shape[0])
        self.flatten_size = 1024
        # this output shape 128x128
        layers = [
            nn.ConvTranspose2d(self.flatten_size, 512, 6, stride=2, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 6, stride=2, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 6, stride=2, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 6, stride=2, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2, padding=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 6, stride=2, padding=2), nn.ReLU(),  # Add this for 256
        ]
        channels = 16
        if self.img_shape[0] == 256:
            layers.append(nn.ConvTranspose2d(16, 8, 6, stride=2, padding=2))
            layers.append(nn.ReLU())
            channels = 8
        layers.append(nn.ConvTranspose2d(channels, 3, 6, stride=2, padding=2))
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)
        # We want to use half of the latent size as the extra length of the aspect ratio
        self.fc = nn.Linear(latent_size, int(self.flatten_size * self.aspect_ratio))

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.reshape(-1, self.flatten_size, 1, self.aspect_ratio)
        out = self.main(x)
        return out


class Map2Camera(nn.Module):
    def __init__(self, map_shape, camera_shape, latent_size=10, attn=False):
        super().__init__()
        self.map_shape = map_shape
        self.camera_shape = camera_shape
        self.encoder = MapEncoder(map_shape, latent_size=latent_size, attn=attn)
        self.decoder = GeneralDecoder(camera_shape, latent_size=latent_size)

    def forward(self, x, meta):
        mu, logsigma, encoder_features = self.encoder(x, meta)
        latent = reparameterize(mu, logsigma)
        converted_x = self.decoder(latent)
        return mu, logsigma, converted_x


class GeneralVAE(nn.Module):
    def __init__(self, source_shape, target_shape, latent_size=20, attn=False):
        super().__init__()
        self.source_shape = source_shape
        self.target_shape = target_shape
        self.latent_size = latent_size
        self.attn = attn
        self.encoder = GeneralEncoder(source_shape, latent_size=latent_size, attn=attn)
        self.decoder = GeneralDecoder4(target_shape, latent_size=latent_size)

    def forward(self, x, meta):
        mu, logsigma, encoder_features, mu_attn_map, logsigma_attn_map = self.encoder(x, meta)
        latent = reparameterize(mu, logsigma)
        converted_x = self.decoder(latent)
        return mu, logsigma, converted_x, mu_attn_map, logsigma_attn_map

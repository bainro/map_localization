import os
from datetime import datetime
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from torch_dataloader import PerspectiveTransformTorchDataset
from utils import parse_args, frange_cycle_sigmoid, LinearScheduler
from vaegan_model import VAEGAN1, VAEGAN_JINWEI

args = parse_args()
writer = SummaryWriter("%s/vaegan/%s_%s" % (args.log_dir, args.tag, datetime.now().strftime("%Y%m%d-%H%M%S")))


seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


alpha = 0.1
gamma = 100
# beta_np_cyc = frange_cycle_sigmoid(0, 1.0, args.epochs, n_cycle=2)
# beta_np_cyc = frange_cycle_sigmoid(0, 1.0, args.epochs, n_cycle=10)
kl_scheduler = LinearScheduler(args.start_time, args.start_value, args.end_time, args.end_value)

# args.img_size = 64
# vaegan = VAEGAN1(img_size=args.img_size).to(device)

# args.img_size = 160
# vaegan = VAEGAN2(img_size=args.img_size).to(device)
# vaegan = VAEGAN_DENSENET(img_size=args.img_size).to(device)

# args.img_size = 128
vaegan = VAEGAN_JINWEI(img_size=args.img_size).to(device)
optimizer_E = optim.Adam(vaegan.encoder.parameters(), lr=args.lr)
optimizer_G = optim.Adam(vaegan.generator.parameters(), lr=args.lr)
optimizer_D = optim.Adam(vaegan.discriminator.parameters(), lr=args.lr * alpha)
scheduler_E = ReduceLROnPlateau(optimizer_E, 'min', patience=5, verbose=True)
scheduler_G = ReduceLROnPlateau(optimizer_G, 'min', patience=5, verbose=True)
scheduler_D = ReduceLROnPlateau(optimizer_D, 'min', patience=5, verbose=True)
bce = nn.BCELoss().to(device)


def get_gan_loss(data, recon_batch, random_fake):
    ones_label = torch.ones(args.batch_size, 1).to(device)
    zeros_label = torch.zeros(args.batch_size, 1).to(device)
    gan_real = bce(vaegan.discriminator(data)[0], ones_label)
    gan_recon = bce(vaegan.discriminator(recon_batch)[0], zeros_label)
    gan_fake = bce(vaegan.discriminator(random_fake)[0], zeros_label)
    # gan_loss = gan_real + gan_recon + gan_fake
    gan_loss = gan_real + gan_recon
    return gan_loss, gan_real, gan_recon, gan_fake


train_dataset = PerspectiveTransformTorchDataset(args.source, square=True, target_size=args.img_size,
                                                 map2camera=args.map2camera, debug=args.debug, debug_size=args.debug_size)
test_dataset = PerspectiveTransformTorchDataset(args.source, train=False, square=True, target_size=args.img_size,
                                                map2camera=args.map2camera, debug=args.debug, debug_size=args.debug_size)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [0.7, 0.3])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          prefetch_factor=16 if args.workers else None, drop_last=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          prefetch_factor=16 if args.workers else None, drop_last=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                         prefetch_factor=16 if args.workers else None, drop_last=True, pin_memory=True)

# Early stopping
best_valid_loss = float('inf')
counter = 0


def evaluate(loader, i_epoch, mode):
    # validation
    vaegan.eval()

    eval_dis_recon_loss = 0
    eval_recon_loss = 0
    eval_kl_loss = 0
    eval_gan_loss = 0
    eval_gan_real = 0
    eval_gan_recon = 0
    eval_gan_fake = 0

    with torch.no_grad():
        for batch_idx, (source_data, target_data, meta) in enumerate(loader):
            source_data = source_data.to(device)
            target_data = target_data.to(device)
            meta = meta.to(device)
            recon_batch, mu, logsigma = vaegan(source_data, meta)
            random_z = torch.randn(args.batch_size, 128).to(device)
            random_fake = vaegan.generator(random_z)
            x_l_tilda = vaegan.discriminator(recon_batch)[1]
            x_l = vaegan.discriminator(target_data)[1]
            dis_recon_loss = ((x_l_tilda - x_l) ** 2).mean()
            gan_loss, gan_real, gan_recon, gan_fake = get_gan_loss(target_data, recon_batch, random_fake)
            kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
            kl_loss = kl_loss / torch.numel(source_data)
            recon_loss = nn.functional.mse_loss(target_data, recon_batch, reduction='mean')
            eval_dis_recon_loss += dis_recon_loss.item()
            eval_recon_loss += recon_loss.item()
            eval_kl_loss += kl_loss.item()
            eval_gan_loss += gan_loss.item()
            eval_gan_real += gan_real.item()
            eval_gan_recon += gan_recon.item()
            eval_gan_fake += gan_fake.item()

    eval_dis_recon_loss /= len(loader)
    eval_recon_loss /= len(loader)
    eval_kl_loss /= len(loader)
    eval_gan_loss /= len(loader)
    eval_gan_real /= len(loader)
    eval_gan_recon /= len(loader)
    eval_gan_fake /= len(loader)

    # tensorboard logging
    writer.add_scalar('%s/dis_recon_loss' % mode, eval_dis_recon_loss, i_epoch)
    writer.add_scalar('%s/recon_loss' % mode, eval_recon_loss, i_epoch)
    writer.add_scalar('%s/kl_loss' % mode, eval_kl_loss, i_epoch)
    writer.add_scalar('%s/gan_loss' % mode, eval_gan_loss, i_epoch)
    writer.add_scalar('%s/gan_real' % mode, eval_gan_real, i_epoch)
    writer.add_scalar('%s/gan_recon' % mode, eval_gan_recon, i_epoch)
    writer.add_scalar('%s/gan_fake' % mode, eval_gan_fake, i_epoch)

    if i_epoch % args.log_freq == 0:
        writer.add_images('%s/images/source' % mode, source_data[:8], i_epoch)
        writer.add_images('%s/images/target' % mode, target_data[:8], i_epoch)
        writer.add_images('%s/images/transform' % mode, recon_batch[:8], i_epoch)

    return eval_recon_loss, eval_gan_loss, eval_dis_recon_loss, eval_kl_loss


for i_epoch in range(args.epochs):
    vaegan.train()
    kl_scheduler.step()
    train_loss = 0
    # beta = beta_np_cyc[i_epoch]
    beta = args.beta * kl_scheduler.val()
    print("kl_beta: ", beta)

    train_dis_recon_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    train_gan_loss = 0
    train_gan_real = 0
    train_gan_recon = 0
    train_gan_fake = 0

    for batch_idx, (source_data, target_data, meta) in enumerate(train_loader):
        source_data = source_data.to(device)
        target_data = target_data.to(device)
        meta = meta.to(device)

        # Train vae.discriminator
        random_z = torch.randn(args.batch_size, 128).to(device)
        random_fake = vaegan.generator(random_z)
        recon_batch, mu, logsigma = vaegan(source_data, meta)
        gan_loss, gan_real, gan_recon, gan_fake = get_gan_loss(target_data, recon_batch, random_fake)

        if not args.disable_gan_loss:
            optimizer_D.zero_grad()
            gan_loss.backward(retain_graph=True)
            optimizer_D.step()

            # Train generator
            gan_loss, gan_real, gan_recon, gan_fake = get_gan_loss(target_data, recon_batch, random_fake)
            x_l_tilda = vaegan.discriminator(recon_batch)[1]
            x_l = vaegan.discriminator(target_data)[1]
            dis_recon_loss = ((x_l_tilda - x_l) ** 2).mean()
            recon_loss = nn.functional.mse_loss(target_data, recon_batch, reduction='mean')
            if args.recon_loss_type == "dis":
                err_gen = gamma * dis_recon_loss - gan_loss
            elif args.recon_loss_type == "pixel":
                err_gen = gamma * recon_loss - gan_loss
            elif args.recon_loss_type == "both":
                err_gen = gamma * dis_recon_loss + gamma * recon_loss - gan_loss
            optimizer_G.zero_grad()
            err_gen.backward(retain_graph=True)
            optimizer_G.step()

        # Train encoder
        recon_batch, mu, logsigma = vaegan(source_data, meta)
        x_l_tilda = vaegan.discriminator(recon_batch)[1]
        x_l = vaegan.discriminator(target_data)[1]
        dis_recon_loss = ((x_l_tilda - x_l) ** 2).mean()
        recon_loss = nn.functional.mse_loss(target_data, recon_batch, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
        kl_loss = kl_loss / torch.numel(source_data)

        if args.recon_loss_type == "dis":
            err_enc = beta * kl_loss + 5 * dis_recon_loss
        elif args.recon_loss_type == "pixel":
            err_enc = beta * kl_loss + 5 * recon_loss
        elif args.recon_loss_type == "both":
            err_enc = beta * kl_loss + 5 * recon_loss + 5 * dis_recon_loss
        optimizer_E.zero_grad()
        if args.disable_gan_loss:
            optimizer_G.zero_grad()
        err_enc.backward(retain_graph=True)
        optimizer_E.step()
        if args.disable_gan_loss:
            optimizer_G.step()

        train_dis_recon_loss += dis_recon_loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        train_gan_loss += gan_loss.item()
        train_gan_real += gan_real.item()
        train_gan_recon += gan_recon.item()
        train_gan_fake += gan_fake.item()

    writer.add_scalar('train/dis_recon_loss', train_dis_recon_loss / len(train_loader), i_epoch)
    writer.add_scalar('train/recon_loss', train_recon_loss / len(train_loader), i_epoch)
    writer.add_scalar('train/kl_loss', train_kl_loss / len(train_loader), i_epoch)
    writer.add_scalar('train/gan_loss', train_gan_loss / len(train_loader), i_epoch)
    writer.add_scalar('train/gan_real', train_gan_real / len(train_loader), i_epoch)
    writer.add_scalar('train/gan_recon', train_gan_recon / len(train_loader), i_epoch)
    writer.add_scalar('train/gan_fake', train_gan_fake / len(train_loader), i_epoch)
    writer.add_scalar('train/kl_annealing', kl_scheduler.val(), i_epoch)

    if i_epoch % args.log_freq == 0:
        writer.add_images('train/images/source', source_data[:8], i_epoch)
        writer.add_images('train/images/target', target_data[:8], i_epoch)
        writer.add_images('train/images/transform', recon_batch[:8], i_epoch)

    valid_recon_loss, valid_gan_loss, valid_dis_recon_loss, valid_kl_loss = evaluate(valid_loader, i_epoch, "valid")
    valid_err_gen = gamma * valid_dis_recon_loss - valid_gan_loss
    valid_err_enc = beta * valid_kl_loss + 5 * valid_dis_recon_loss
    scheduler_D.step(valid_recon_loss)
    scheduler_G.step(valid_recon_loss)
    scheduler_E.step(valid_recon_loss)

    print('====> Epoch: {} valid recon loss: {:.4f}'.format(i_epoch, valid_recon_loss))

    if valid_recon_loss < best_valid_loss:
        best_valid_loss = valid_recon_loss
        counter = 0
        print("Saving model, valid_loss: %.4f" % valid_recon_loss)
        torch.save(vaegan.state_dict(), '%s/vaegan.pt' % writer.get_logdir())
    else:
        counter += 1
        if counter >= args.es_patience:
            print("Early stopping...")
            break

# Test
vaegan.load_state_dict(torch.load('%s/vaegan.pt' % writer.get_logdir()))
test_recon_loss, _, _, _ = evaluate(test_loader, 0, "test")
print('====> Test recon loss: {:.4f}'.format(test_recon_loss))

writer.close()

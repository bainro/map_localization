import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision.transforms as T
import numpy as np
import random

from vae_model import GeneralVAE
from torch_dataloader import PerspectiveTransformTorchDataset
from utils import parse_args

args = parse_args()

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def vae_loss(x, mu, logsigma, recon_x, beta=1):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl_loss = kl_loss / torch.numel(x)
    return recon_loss + kl_loss * beta, recon_loss, kl_loss


def evaluate(model, loader, kl_scheduler, writer, i_epoch, name):
    model.eval()
    all_loss = []
    all_recon_loss = []
    all_kl_loss = []
    for i_batch, (source_data, target_data, meta) in enumerate(loader):
        source_data = source_data.to(device)
        target_data = target_data.to(device)
        meta = meta.to(device)
        mu, logsigma, predict_target, mu_attn_map, logsigma_attn_map = model(source_data, meta)
        loss, recon_loss, kl_loss = vae_loss(target_data, mu, logsigma, predict_target, args.beta * kl_scheduler.val())
        all_loss.append(loss.detach().cpu())
        all_recon_loss.append(recon_loss.detach().cpu())
        all_kl_loss.append(kl_loss.detach().cpu())

    mean_loss = torch.stack(all_loss).mean()
    mean_recon_loss = torch.stack(all_recon_loss).mean()
    mean_kl_loss = torch.stack(all_kl_loss).mean()

    print("Epoch %i %s loss: %.4f, recon_loss: %.4f, kl_loss: %.4f" % (
    i_epoch, name, mean_loss, mean_recon_loss, mean_kl_loss))
    writer.add_scalar('%s/loss' % name, mean_loss.detach().cpu().item(), i_epoch)
    writer.add_scalar('%s/recon_loss' % name, mean_recon_loss.detach().cpu().item(), i_epoch)
    writer.add_scalar('%s/kl_loss' % name, mean_kl_loss.detach().cpu().item(), i_epoch)

    if i_epoch % args.log_freq == 0:
        writer.add_images('%s/images/source' % name, source_data[:8], i_epoch)
        writer.add_images('%s/images/target' % name, target_data[:8], i_epoch)
        writer.add_images('%s/images/transform' % name, predict_target[:8], i_epoch)
    return mean_loss, mean_recon_loss, mean_kl_loss, source_data, target_data, predict_target


def plot_attn_map(axs, source_img, attn_maps, name=""):
    transform = T.Resize(size=args.batch_size)
    attn_maps = attn_maps.reshape(args.latent_size, 15, 15)
    attn_maps = transform(attn_maps)
    for j, (ax, attn) in enumerate(zip(axs.flatten(), attn_maps)):
        ax.imshow(source_img.transpose(0, 2).detach().cpu())
        ax.imshow(attn.detach().cpu(), cmap='jet', alpha=0.2)
        ax.set_title("%s_%i" %(name, j))
        ax.axis('off')


def main():
    ########### Preparation ##########
    # 1. dataset and dataloader
    train_dataset = PerspectiveTransformTorchDataset(args.source, target_size=args.img_size, square=not args.not_square,
                                                     map2camera=args.map2camera, debug=args.debug)
    test_dataset = PerspectiveTransformTorchDataset(args.source, train=False, target_size=args.img_size, square=not args.not_square,
                                                    map2camera=args.map2camera, debug=args.debug)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [0.7, 0.3])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              prefetch_factor=16 if args.workers else None, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              prefetch_factor=16 if args.workers else None, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                             prefetch_factor=16 if args.workers else None, drop_last=True, pin_memory=True)

    source_shape, target_shape = test_dataset.get_shape()
    # 2. model and optimizer
    model = GeneralVAE(source_shape, target_shape, latent_size=args.latent_size, attn=args.attn).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ########### Train ############
    # 3. log
    writer = SummaryWriter("%s/vae/%s_%s" % (args.log_dir, args.tag, datetime.now().strftime("%Y%m%d-%H%M%S")))
    save_args(args, writer.log_dir)

    # 4. main train
    step = 0
    kl_scheduler = LinearScheduler(args.start_time, args.start_value, args.end_time, args.end_value)
    best_valid_recon_loss = float("Inf")
    patient = 0

    for i_epoch in range(args.epochs):
        model.train()
        kl_scheduler.step()
        for i_batch, (source_data, target_data, meta) in enumerate(train_loader):
            source_data = source_data.to(device)
            target_data = target_data.to(device)
            meta = meta.to(device)
            mu, logsigma, predict_target, mu_attn_map, logsigma_attn_map = model(source_data, meta)
            loss, recon_loss, kl_loss = vae_loss(target_data, mu, logsigma, predict_target, args.beta * kl_scheduler.val())

            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1

        valid_loss, valid_recon_loss, valid_kl_loss, _, _, _ = evaluate(model, valid_loader, kl_scheduler, writer, i_epoch, "valid")
        if valid_recon_loss < best_valid_recon_loss:
            print("Saving model, recon_loss: %.4f" % valid_recon_loss)
            torch.save(model.state_dict(), '%s/best_model.pt' % writer.get_logdir())
            patient = 0
            best_valid_recon_loss = valid_recon_loss
        else:
            patient += 1
            if patient > args.es_patience:
                print("Valid loss increases, early stopping.")
                break

        # # Reload best model
        # model.load_state_dict(torch.load('%s/best_model.pt' % writer.get_logdir()))

        # Test set
        test_loss, test_recon_loss, test_kl_loss, test_source_data, test_target_data, test_predict_target = evaluate(model, test_loader, kl_scheduler, writer, 0, "test")
        print("test loss: %.4f, test recon_loss: %.4f, test kl_loss: %.4f" % (
            test_loss, test_recon_loss, test_kl_loss))


if __name__ == '__main__':
    main()

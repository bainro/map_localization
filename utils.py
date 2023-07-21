import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--latent-size', type=int, default=100)
    parser.add_argument('--map2camera', default=False, action='store_true')
    parser.add_argument('--log-dir', type=str, default="runs")
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--log-freq', type=int, default=20)
    parser.add_argument('--es-patience', type=int, default=100)
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--not-square', default=False, action="store_true")
    parser.add_argument('--debug-size', type=int, default=100)
    parser.add_argument('--attn', default=False, action='store_true')
    parser.add_argument('--att-weight', type=float, default=100)
    parser.add_argument('--recon-loss-type', type=str, default="dis")
    parser.add_argument('--disable-gan-loss', default=False, action="store_true")

    # KL annealing parameters
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--start-time', type=int, default=50)
    parser.add_argument('--start-value', type=int, default=0)
    parser.add_argument('--end-time', type=int, default=1000)
    parser.add_argument('--end-value', type=int, default=1)

    args = parser.parse_args()

    if args.debug:
        args.log_dir = "debug_runs"
        args.workers = 0
        args.batch_size = 8
    return args

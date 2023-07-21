import argparse
import os
import json

import dateutil.parser
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
from scipy.spatial.transform import Rotation


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


def parse_phone_gps(source):
    tree = ET.parse(source)
    root = tree.getroot()
    ns = {'p': 'http://www.topografix.com/GPX/1/0'}
    xml_tags = root.findall('p:trk/p:trkseg/p:trkpt', ns)
    phone_gps_log = []
    for tag in xml_tags:
        gps = tag.attrib
        gps['lat'] = float(gps['lat'])
        gps['long'] = float(gps['lon'])
        gps['time'] = dateutil.parser.isoparse(tag.find('p:time', ns).text)
        gps['speed'] = float(tag.find('p:speed', ns).text)
        phone_gps_log.append(gps)
    return pd.DataFrame(phone_gps_log).set_index("time")


# Filter out shaky frames
def not_shaky(frame, threshold=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    flow = cv2.calcOpticalFlowFarneback(gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    norm = cv2.norm(flow, cv2.NORM_L2)
    return norm / (h * w) < threshold


# Filter out blurry frames
def not_blurry(frame, not_blury_threshold=70):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm > not_blury_threshold


def not_similar(previous_frame, frame, diff_threshold=0.2):
    if previous_frame is None:
        return True
    # print("pixel diff:", (cv2.absdiff(previous_frame, frame) / 255 > 0.1).mean(), "structural diff:", 1 - structural_similarity(previous_frame, frame, channel_axis=2))
    if (cv2.absdiff(previous_frame, frame) / 255 > 0.1).mean() < diff_threshold:
        return False
    elif 1 - structural_similarity(previous_frame, frame, channel_axis=2, multichannel=True) < diff_threshold:
        return False
    else:
        return True


def get_heading(imu):
    # quarternion format
    rot = Rotation.from_quat(imu)
    _roll, _pitch, yaw = rot.as_euler('xyz', degrees=True)
    heading = yaw
    heading *= -1
    heading += 360
    if heading > 360:
        heading = heading - 360
    return heading


class LinearScheduler:
    def __init__(self, start_time=0, start_value=1, end_time=1, end_value=1):
        self.t = 0
        self.start_time, self.start_value = start_time, start_value
        self.end_time, self.end_value = end_time, end_value

    def step(self):
        self.t += 1

    def val(self):
        return self.start_value + \
               (self.t < self.end_time and self.t >= self.start_time) * (self.t - self.start_time) / (
                           self.end_time - self.start_time) * (self.end_value - self.start_value) + \
               (self.t >= self.end_time) * (self.end_value - self.start_value)


def save_args(args, save_path, name='config.json'):
    '''input: Argument Parser object and save path'''

    save_object = vars(args)
    save_path = os.path.join(save_path, name) if os.path.isdir(save_path) else save_path
    with open(save_path, 'w+') as f:
        json.dump(save_object, f, indent=4)


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    # https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop:
            L[int(i + c * period)] = 1.0 / (1.0 + np.exp(- (v * 12. - 6.)))
            v += step
            i += 1
    return L


if __name__ == "__main__":
    n_epoch = 1000
    beta_np_cyc = frange_cycle_sigmoid(0.0, 1.0, n_epoch, 2)
    beta_np_inc = frange_cycle_sigmoid(0.0, 1.0, n_epoch, 1, 0.25)
    beta_np_con = np.ones(n_epoch)

    fig = plt.figure(figsize=(8, 4.0))
    stride = max(int(n_epoch / 8), 1)

    plt.plot(range(n_epoch), beta_np_cyc, '-', label='Cyclical', marker='s', color='k', markevery=stride, lw=2, mec='k',
             mew=1, markersize=10)
    plt.plot(range(n_epoch), beta_np_inc, '-', label='Monotonic', marker='o', color='r', markevery=stride, lw=2,
             mec='k', mew=1, markersize=10)

    leg = plt.legend(fontsize=16, shadow=True, loc=(0.65, 0.2))
    plt.grid(True)

    plt.xlabel('# Iteration', fontsize=16)
    plt.ylabel("$\\beta$", fontsize=16)

    ax = plt.gca()

    # Left Y-axis labels
    plt.yticks((0.0, 0.5, 1.0), ('0', '0.5', '1'), color='k', size=14)

    plt.xlim(0, n_epoch)
    plt.ylim(-0.1, 1.1)

    plt.show()

import argparse
import concurrent.futures
from datetime import datetime, timedelta
import glob
import os
import shutil
import pytz
import io
import math

import cv2
import folium
import pandas as pd
from PIL import Image
import geopy.distance
import numpy as np

import utils


skip_interval = 0
distance_threshold = 10
meters_per_pixel = 0.49436355985590763  # for zoom level 19
aldrich_center = [33.64594695480951, -117.84275532892153]
map_size = 400
map_service = 'OpenStreetMap'
use_satellite = True

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default="data")
parser.add_argument('--redo', default=False, action="store_true")
parser.add_argument('--debug', default=False, action="store_true")
args = parser.parse_args()


def extract_frames(source_dir):
    if "video.avi" in os.listdir(source_dir):
        return extract_jackal_frames(source_dir)
    elif len(glob.glob(source_dir + "/*.mp4")):
        return extract_phone_360_frames(source_dir)
    elif len(glob.glob(source_dir + "/*.mkv")):
        return extract_hsr_frames(source_dir)


def extract_jackal_frames(source_dir):
    print(source_dir)
    # Load video
    video = cv2.VideoCapture(os.path.join(source_dir, "video.avi"))
    frame_log = pd.read_csv(os.path.join(source_dir, "frame_log.txt"), header=None)
    gps_log = pd.read_csv(os.path.join(source_dir, "gps_log.txt"), index_col=0)
    phone_gps_file = os.path.join(source_dir, "phone_gps.gpx")
    if os.path.exists(phone_gps_file):
        phone_gps_log = utils.parse_phone_gps(phone_gps_file)
    else:
        phone_gps_log = None
    imu_log_file = os.path.join(source_dir, "imu_log.txt")
    if os.path.exists(imu_log_file):
        imu_log = pd.read_csv(imu_log_file, index_col=0)
    else:
        imu_log = None

    destination_dir = os.path.join(source_dir, "extracted_frames")
    if os.path.exists(destination_dir):
        if args.redo:
            shutil.rmtree(destination_dir)
        else:
            print("Skipping %s as it is already extracted" % source_dir)
            return 0
    os.makedirs(destination_dir, exist_ok=True)
    meta_data_file = open(os.path.join(destination_dir, "meta_data.csv"), "w")
    meta_data_file.write("frame,time,heading\n")

    # Extract frames
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("frame_count: %i" % frame_count)

    previous_frame = None
    extracted_frames = 0

    for i in range(frame_count):
        ret, frame = video.read()
        if ret:
            if utils.not_blurry(frame) and utils.not_similar(previous_frame, frame):
                estimate_loc = int(len(frame_log) * i / frame_count)
                timestamp = frame_log.iloc[estimate_loc]
                parsed_datetime = datetime.fromtimestamp(int(timestamp), tz=pytz.UTC)
                try:
                    gps = gps_log.loc[timestamp]
                    if len(gps.shape) > 1:
                        gps = gps.mean(0)
                except KeyError:
                    print("Frame: %i cannot find gps" % i)
                    continue

                coord = [gps['lat'], gps['long']]
                if phone_gps_log is not None:
                    try:
                        phone_gps = phone_gps_log.loc[parsed_datetime]
                        phone_coord = [phone_gps['lat'], phone_gps['long']]
                        distance = geopy.distance.geodesic(coord, phone_coord).m
                        print("Frame: %i, jackal gps: %r, phone gps: %r, distance: %.2fm" %
                              (i, coord, phone_coord, distance))
                        if distance > distance_threshold:
                            print("GPS coordinate differed too much!!")
                            continue
                    except Exception:
                        # If anything is not working, just skip this frame
                        continue

                if imu_log is not None:
                    try:
                        imu = imu_log.loc[timestamp].mean(0)
                        heading = utils.get_heading(imu)
                    except Exception:
                        # If anything is not working, just skip this frame
                        continue

                center = coord
                aldrich_park_map = folium.Map(location=center, zoom_start=18, width=map_size,
                                              height=map_size,
                                              tiles=map_service)
                v_dist = geopy.distance.geodesic([coord[0], center[1]], center).m * math.copysign(1, coord[0] - center[0])
                h_dist = geopy.distance.geodesic([center[0], coord[1]], center).m * math.copysign(1, coord[1] - center[1])

                # v_pixel = v_dist / meters_per_pixel  # image original is top left corner
                # h_pixel = h_dist / meters_per_pixel
                print("Frame: %i, v_dist: %f, h_dist: %f, heading: %.2f" %
                      (i, v_dist, h_dist, heading))
                irvine_time = parsed_datetime.astimezone(pytz.timezone('US/Pacific'))
                meta_data_file.write("%s,%s,%.2f\n" % (i, irvine_time.strftime('%H'), heading))
                # CircleMarker with radius
                marker = folium.Circle(location=coord, radius=5, fill=True, fillOpacity=1)
                marker.add_to(aldrich_park_map)
                map_img_data = aldrich_park_map._to_png(5)
                map_img = Image.open(io.BytesIO(map_img_data))
                map_img.save(os.path.join(destination_dir, "%i_map.png" % i))
                cv2.imwrite(os.path.join(destination_dir, "%i_camera.png" % i), frame)
                previous_frame = frame
                extracted_frames += 1
                if args.debug and extracted_frames >= 500:
                    break
        else:
            break
    meta_data_file.close()
    return extracted_frames


def extract_phone_360_frames(source_dir):
    print(source_dir)
    # Load video
    video_file = glob.glob(source_dir + "/*.mp4")[0]
    base_filename = os.path.splitext(os.path.basename(video_file))[0]
    not_blurry_th = 70
    diff_th = 0.2
    aldrich_satellite = Image.open('aldrich_satellite.png')
    aldrich_satellite_meters_per_pixel = 1200 / aldrich_satellite.width     # I cropped a square of 1.2km wide

    if "VID" in base_filename:
        base_filename = base_filename.split('_')[1] + '_' + base_filename.split('_')[2]
        not_blurry_th = 50
    video_start_time = pytz.timezone('US/Pacific').localize(datetime.strptime(base_filename, '%Y%m%d_%H%M%S'))

    video = cv2.VideoCapture(video_file)
    gps_log = pd.read_csv(os.path.join(source_dir, "Location.csv"), index_col=0)
    gps_log.index = (gps_log.index / 1e9).astype(int)

    destination_dir = os.path.join(source_dir, "extracted_frames")
    if os.path.exists(destination_dir):
        if args.redo:
            shutil.rmtree(destination_dir)
        else:
            print("Skipping %s as it is already extracted" % source_dir)
            return 0

    os.makedirs(destination_dir, exist_ok=True)
    meta_data_file = open(os.path.join(destination_dir, "meta_data.csv"), "w")
    meta_data_file.write("frame,time,heading\n")

    # Extract frames
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("frame_count: %i" % frame_count)

    previous_frame = None
    previous_frame_ix = 0
    extracted_frames = 0

    for i in range(frame_count):
        ret, frame = video.read()
        if ret:
            # print("Frame: %i,  not_blurry: %r, not_similar: %r, frame delta: %i (%r)" %
            #       (i, utils.not_blurry(frame, not_blurry_th), utils.not_similar(previous_frame, frame, diff_th), i - previous_frame_ix, i - previous_frame_ix >= skip_interval))
            if utils.not_blurry(frame, not_blurry_th) and utils.not_similar(previous_frame, frame, diff_th) \
                    and i - previous_frame_ix >= skip_interval:
                # print("cv2.CAP_PROP_POS_MSEC:", video.get(cv2.CAP_PROP_POS_MSEC))
                parsed_datetime = video_start_time + timedelta(milliseconds=video.get(cv2.CAP_PROP_POS_MSEC))
                try:
                    gps = gps_log.loc[int(parsed_datetime.timestamp())]
                    if len(gps.shape) > 1:
                        gps = gps.mean(0)
                except KeyError:
                    print("Frame: %i cannot find gps" % i)
                    continue

                coord = [gps['latitude'], gps['longitude']]
                heading = gps['bearing']
                irvine_time = parsed_datetime.astimezone(pytz.timezone('US/Pacific'))
                meta_data_file.write("%s,%s,%.2f\n" % (i, irvine_time.strftime('%H'), heading))
                print("Frame: %i, heading: %.2f, total: %i" % (i, heading, extracted_frames))

                if use_satellite:
                    v_dist = geopy.distance.geodesic([coord[0], aldrich_center[1]],
                                                     aldrich_center).m * math.copysign(1, coord[0] - aldrich_center[0])
                    h_dist = geopy.distance.geodesic([aldrich_center[0], coord[1]],
                                                     aldrich_center).m * math.copysign(1, coord[1] - aldrich_center[1])
                    v_pixel = v_dist / aldrich_satellite_meters_per_pixel
                    h_pixel = h_dist / aldrich_satellite_meters_per_pixel
                    center_h = aldrich_satellite.width // 2 + h_pixel  # image original is top left corner
                    center_v = aldrich_satellite.height // 2 - v_pixel  # image original is top left corner
                    satelite_img = aldrich_satellite.crop(((center_h - map_size // 2,
                                        center_v - map_size // 2,
                                        center_h + map_size // 2,
                                        center_v + map_size // 2)))
                    satelite_img.save(os.path.join(destination_dir, "%i_map.png" % i))

                else:
                    aldrich_park_map = folium.Map(location=coord, zoom_start=18, width=map_size,
                                                  height=map_size, tiles=map_service)

                    # CircleMarker with radius
                    marker = folium.Circle(location=coord, radius=5, fill=True, fillOpacity=1)
                    marker.add_to(aldrich_park_map)
                    map_img_data = aldrich_park_map._to_png(5)
                    map_img = Image.open(io.BytesIO(map_img_data))
                    map_img.save(os.path.join(destination_dir, "%i_map.png" % i))
                cv2.imwrite(os.path.join(destination_dir, "%i_camera.png" % i), frame)
                previous_frame = frame
                extracted_frames += 1
                previous_frame_ix = i
                if args.debug and extracted_frames >= 500:
                    break
        else:
            break
    meta_data_file.close()
    return extracted_frames


def extract_hsr_frames(source_dir):
    print(source_dir)
    # Load video
    video_file = glob.glob(source_dir + "/*.mkv")[0]
    base_filename = os.path.splitext(os.path.basename(video_file))[0]
    not_blurry_th = 120
    diff_th = 0.2
    aldrich_satellite = Image.open('aldrich_satellite.png')
    aldrich_satellite_meters_per_pixel = 1200 / aldrich_satellite.width     # I cropped a square of 1.2km wide

    video_start_time = pytz.timezone('US/Pacific').localize(datetime.strptime(base_filename, '%Y-%m-%d %H-%M-%S'))
    video = cv2.VideoCapture(video_file)

    destination_dir = os.path.join(source_dir, "extracted_frames")
    if os.path.exists(destination_dir):
        if args.redo:
            shutil.rmtree(destination_dir)
        else:
            print("Skipping %s as it is already extracted" % source_dir)
            return 0

    os.makedirs(destination_dir, exist_ok=True)
    meta_data_file = open(os.path.join(destination_dir, "meta_data.csv"), "w")
    meta_data_file.write("frame,time,heading\n")

    # Extract frames
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("frame_count: %i" % frame_count)

    previous_frame = None
    previous_frame_ix = 0
    extracted_frames = 0

    for i in range(frame_count):
        ret, frame = video.read()
        if ret:
            # Check if there is a gray patch at this location so that we know it is running rvix
            gray_patch = frame[37:47, 140:675]
            is_gray = np.isclose(gray_patch.mean(), 237, atol=3)
            if is_gray:
                map_view = frame[76:336, 54:390]
                camera_view = frame[107:317, 406:679]

                if utils.not_blurry(frame, not_blurry_th) and utils.not_similar(previous_frame, camera_view, diff_th) \
                    and i - previous_frame_ix >= skip_interval:
                    parsed_datetime = video_start_time + timedelta(milliseconds=video.get(cv2.CAP_PROP_POS_MSEC))
                    irvine_time = parsed_datetime.astimezone(pytz.timezone('US/Pacific'))
                    heading = 0
                    meta_data_file.write("%s,%s,%.2f\n" % (i, irvine_time.strftime('%H'), heading))
                    # map_img = Image.open(io.BytesIO(map_img_data))
                    # map_img.save(os.path.join(destination_dir, "%i_map.png" % i))
                    cv2.imwrite(os.path.join(destination_dir, "%i_map.png" % i), map_view)
                    cv2.imwrite(os.path.join(destination_dir, "%i_camera.png" % i), camera_view)
                    previous_frame = camera_view
                    extracted_frames += 1
                    previous_frame_ix = i
                    print("Extracted frame %i, total %i frames" % (i, extracted_frames))
                    if args.debug and extracted_frames >= 500:
                        break
        else:
            break
    meta_data_file.close()
    return extracted_frames


if __name__ == "__main__":
    valid_dir = [os.path.join(args.source, dir) for dir in os.listdir(args.source) if
                 os.path.isdir(os.path.join(args.source, dir)) and not dir.startswith('.')]
    if args.debug:
        Executor = concurrent.futures.ThreadPoolExecutor
        # valid_dir = valid_dir[1:2]
    else:
        Executor = concurrent.futures.ProcessPoolExecutor
    with Executor() as executor:
        total_extracted = executor.map(extract_frames, valid_dir)
    print("Extracted %i frames in total" % sum(total_extracted))
    args.save_dir = "numpy_data/aldrich_park_not_shuffle"

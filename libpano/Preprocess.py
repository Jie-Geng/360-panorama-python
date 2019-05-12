import os
import math
from multiprocessing import Pool

import numpy as np
import cv2 as cv
import imutils

from libpano import utils
from libpano import Config
from libpano import ImageCropper
from libpano.FocalCalculator import FocalCalculator


def preprocess_resize(image_folder, temp_folder, meta, scale):
    """
    Preprocess Step 1: rotate and resize

    :param image_folder: the folder path containing source images
    :param temp_folder: the folder path to store the processed images
    :param meta: meta data. It is needed to extract file names and roll information.
    :param scale: Resizing scale.
    :return: Nothing, because we save the processed images.
    """
    meta_data = meta.grid_data
    # rows = meta_data.row.nunique()
    cols = meta_data.col.nunique()

    # preparing arguments for multiprocess pool.
    args = []
    for row in range(Config.start_row, Config.end_row):
        for col in range(cols):
            item = meta_data[(meta_data.row == row) & (meta_data.col == col)]
            fn = item.uri.values[0]
            roll = item.roll.values[0]
            arg = (
                os.path.join(image_folder, fn),
                os.path.join(temp_folder, fn),
                scale,
                roll
            )
            args.append(arg)

    # Do it on every core.
    pool = Pool(processes=None)
    pool.map(resize_rotate_frame, args)


def resize_rotate_frame(args):
    """
    Scale down and rotate a frame.

    If we rotate a image after resizing, its dimension would be smaller than we expect,
    due to the cropping after resize.

    So, it is better to precede rotation before resizing.

    :param args: (image_path, save_path, scale, roll)
    :return: nothing
    """

    (src_path, dest_path, scale, roll) = args

    # read and rotate image
    img = cv.imread(src_path)
    orig_height, orig_width, _ = img.shape
    img = imutils.rotate_bound(img, roll)

    # crop image boundaries
    cropper = ImageCropper.ImageCropper(img)
    img = cropper.crop()

    # calculate the new size
    height = int(orig_height * scale)
    width = int(orig_width * scale)

    # resize the image
    img = cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR_EXACT)
    cv.imwrite(dest_path, img)

    # free the memory
    del img


def preprocess_warp(image_folder, meta):
    """
    Preprocess Step w: warping

    From experiments, it's better to perform perspective warping first and then cylindrical warping.

    :param image_folder: the folder path containing images.
    :param meta: meta data.
    :return: Nothing as we save the result to a file
    """

    meta_data = meta.grid_data
    metrics = meta.metrics
    # rows = meta_data.row.nunique()
    cols = meta_data.col.nunique()

    args = []

    # generate parameters for pooling
    for row in range(Config.start_row, Config.end_row):
        for col in range(cols):
            item = meta_data[(meta_data.row == row) & (meta_data.col == col)]
            fn = item.uri.values[0]
            arg = (
                os.path.join(image_folder, fn),
                item.pitch.values[0],
                metrics
            )

            args.append(arg)

    # multi-core process of transform
    pool = Pool(processes=None)
    pool.map(warp_frame, args)


def warp_frame(args):
    """
    Performs Perspective and Cylindrical warping.

    :param args: warp parameters.
    :return: Nothing
    """

    (src_path, pitch, PM) = args
    img = cv.imread(src_path)

    pitch = utils.degree2radian(pitch)

    if Config.debug:
        print('')
        print(os.path.basename(src_path), '\n  pitch={}'.format(pitch))

    cx = PM.FW // 2
    cy = PM.FH // 2

    # calculate the top and bottom pitch
    pitch_top = pitch - PM.AOV_v / 2
    pitch_bottom = pitch + PM.AOV_v / 2

    if pitch_top < -Config.max_vertical_angle:
        pitch_top = -Config.max_vertical_angle

    if pitch_bottom > Config.max_vertical_angle:
        pitch_bottom = Config.max_vertical_angle

    if Config.debug:
        print('  pitch_top={},   pitch_bottom={}'.format(pitch_top, pitch_bottom))

    # calculate delta values of corners of ROI area
    dy_top = int((pitch - pitch_top) * PM.PPR_v)
    dy_bottom = int((pitch_bottom - pitch) * PM.PPR_v)
    dx_top = int(PM.R_h * math.cos(pitch_top) * PM.AOV_h / 2)
    dx_bottom = int(PM.R_h * math.cos(pitch_bottom) * PM.AOV_h / 2)

    # prevent errors
    dy_top = min(dy_top, cy)
    dy_bottom = min(dy_bottom, cy)
    dx_top = min(dx_top, cx)
    dx_bottom = min(dx_bottom, cx)

    if Config.debug:
        print('  dx_top={}, dx_bottom={}, dy_top={}, dy_bottom={}'.format(dx_top, dx_bottom, dy_top, dy_bottom))

    src_pts = np.array([[cx-dx_top, cy-dy_top], [cx+dx_top, cy-dy_top],
                        [cx+dx_bottom, cy+dy_bottom], [cx-dx_bottom, cy+dy_bottom]], np.int32)
    dst_pts = np.array([[0, cy-dy_top], [PM.FH, cy-dy_top], [PM.FH, cy+dy_bottom], [0, cy+dy_bottom]], np.int32)

    # EXPERIMENT
    pts = src_pts.reshape((-1, 1, 2))
    cv.polylines(img, [pts], True, (255, 255, 255), 2)

    h, _ = cv.findHomography(src_pts, dst_pts)
    perspective = cv.warpPerspective(img, h, (PM.FW, PM.FH))

    # EXPERIMENT
    # cv.polylines(img, [pts], True, (255, 0, 0), 2)

    # Cylindrical warp
    k = np.array([[PM.focal_length_px, 0,                  PM.FW / 2],
                  [0,                  PM.focal_length_px, PM.FH / 2],
                  [0,                  0,                  1]], np.float32)

    cylindrical = FocalCalculator.cylindrical_warp(perspective, k)

    # Crop image
    # x = cx - dx - Config.frame_margin
    # y = cy - dy - Config.frame_margin
    # w = 2 * (dx + Config.frame_margin)
    # h = 2 * (dy + Config.frame_margin)
    #
    # img = img[y:y+h, x:x+w]

    # cropper = ImageCropper.ImageCropper(img)
    # img = cropper.crop()

    cv.imwrite(src_path, cylindrical)

    if Config.debug:
        cv.imshow('orig', img)
        cv.imshow('perspective', perspective)
        cv.imshow('cylindrical', cylindrical)
        cv.waitKey(0)
        cv.destroyAllWindows()

    del PM, img, perspective, cylindrical

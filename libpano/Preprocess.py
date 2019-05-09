import os
import math
from multiprocessing import Pool

import numpy as np
import cv2 as cv
import imutils

from libpano import utils
from libpano import Config


def preprocess_resize(image_folder, temp_folder, meta, scale):
    meta_data = meta.grid_data
    # rows = meta_data.row.nunique()
    cols = meta_data.col.nunique()

    args = []
    for row in range(Config.start_row, Config.end_row):
        for col in range(cols):
            item = meta_data[(meta_data.row == row) & (meta_data.col == col)]
            fn = item.uri.values[0]
            arg = (
                os.path.join(image_folder, fn),
                os.path.join(temp_folder, fn),
                scale
            )
            args.append(arg)

    pool = Pool(processes=None)
    pool.map(process_resize, args)


def process_resize(args):
    (src_path, dest_path, scale) = args
    img = cv.imread(src_path)
    img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR_EXACT)
    cv.imwrite(dest_path, img)
    del img


def preprocess_frames(image_folder, meta):
    meta_data = meta.grid_data
    metrics = meta.metrics
    # rows = meta_data.row.nunique()
    cols = meta_data.col.nunique()

    args = []

    # generate parameters for pool
    for row in range(Config.start_row, Config.end_row):
        for col in range(cols):
            item = meta_data[(meta_data.row == row) & (meta_data.col == col)]
            fn = item.uri.values[0]
            arg = (
                os.path.join(image_folder, fn),
                item.roll.values[0],
                item.pitch.values[0],
                item.yaw.values[0],
                metrics
            )

            args.append(arg)

    # multi-core process of transform
    pool = Pool(processes=None)
    pool.map(process_frame, args)

    # process_frame(args[0])


def process_frame(args):
    (src_path, roll, pitch, yaw, PM) = args
    img = cv.imread(src_path)

    pitch = utils.degree2radian(pitch)

    img = imutils.rotate_bound(img, roll)

    cx = PM.FW // 2
    cy = PM.FH // 2

    dy = int(PM.R_h * PM.FIA_v / 2)
    p = pitch - PM.FIA_v / 2
    dx_top = int(PM.R_h * math.cos(p) * PM.FIA_h / 2)
    p += PM.FIA_v
    dx_bottom = int(PM.R_h * math.cos(p) * PM.FIA_h / 2)
    dx = max(dx_top, dx_bottom)

    src_pts = np.array([[cx-dx_top, cy-dy], [cx+dx_top, cy-dy], [cx+dx_bottom, cy+dy], [cx-dx_bottom, cy+dy]], np.int32)
    dst_pts = np.array([[cx-dx, cy-dy], [cx+dx, cy-dy], [cx+dx, cy+dy], [cx-dx, cy+dy]], np.int32)

    # EXPERIMENT
    # pts = src_pts.reshape((-1, 1, 2))
    # cv.polylines(img, [pts], True, (255, 255, 255), 2)

    h, _ = cv.findHomography(src_pts, dst_pts)
    img = cv.warpPerspective(img, h, (PM.FW, PM.FH))

    # EXPERIMENT
    # cv.polylines(img, [pts], True, (255, 0, 0), 2)

    # Crop image
    x = cx - dx - Config.frame_margin
    y = cy - dy - Config.frame_margin
    w = 2 * (dx + Config.frame_margin)
    h = 2 * (dy + Config.frame_margin)

    img = img[y:y+h, x:x+w]

    cv.imwrite(src_path, img)

    del PM, img

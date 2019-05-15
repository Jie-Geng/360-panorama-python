import os

import numpy as np
import cv2 as cv
from math import pi
from multiprocessing import Pool

from libpano import Config
from libpano import utils
from libpano import ImageFrame


def preprocess_frame(args):
    frame, metrics = args
    frame.preprocess_image(1, metrics)
    print(frame.mask.shape)

    return frame


class Stitcher:

    def __init__(self, folder, meta):
        self.folder = folder
        self.meta = meta.grid_data
        self.metrics = meta.metrics

        self.frames = []

        self.blender = cv.detail_MultiBandBlender()
        self.seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR")
        self.corners = []
        self.sizes = []

    def load_and_preprocess(self):

        cols = self.meta.col.nunique()

        for row in range(Config.start_row, Config.end_row):
            for col in range(cols):
                item = self.meta[(self.meta.row == row) & (self.meta.col == col)]
                fn = item.uri.values[0]
                fn = os.path.join(self.folder, fn)

                pitch = item.pitch.values[0]
                yaw = item.yaw.values[0]
                roll = item.roll.values[0]

                frame = ImageFrame.ImageFrame(fn,
                                              utils.degree2radian(pitch),
                                              utils.degree2radian(yaw),
                                              roll)
                self.frames.append(frame)

        # Do it on every core.
        pool = Pool(processes=None)
        frames = pool.map(preprocess_frame, [(frame, self.metrics) for frame in self.frames])
        self.frames = frames

    def position_frames(self):

        ppr_x = Config.internal_panorama_width / (2 * pi)
        ppr_y = Config.internal_panorama_height / pi

        for frame in self.frames:
            print(frame.filename, frame.contents.shape)
            x = frame.yaw * ppr_x
            y = frame.pitch * ppr_y

            x -= (frame.width / 2)
            y -= (frame.height / 2)

            self.corners.append((int(x), int(y)))

    def blend_frames(self):
        self.blender.setNumBands((np.log(Config.frame_margin)/np.log(2.) - 1.).astype(np.int))

        dest_size = cv.detail.resultRoi(corners=self.corners,
                                        sizes=[(frame.width, frame.height) for frame in self.frames])
        self.blender.prepare(dest_size)

        for idx, frame in enumerate(self.frames):
            print('   ', frame.filename)
            print(frame.mask.shape)
            self.blender.feed(cv.UMat(frame.contents), frame.mask, self.corners[idx])

        result = None
        result_mask = None
        result, result_mask = self.blender.blend(result, result_mask)

        return result

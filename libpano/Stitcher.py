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

        self.seam_masks = []

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
            x = frame.yaw * ppr_x
            y = frame.pitch * ppr_y

            x -= (frame.width / 2)
            y -= (frame.height / 2)

            self.corners.append((int(x), int(y)))

    def seam_find(self):
        seam_finder = cv.detail_DpSeamFinder("COLOR_GRAD")

        seam_images = []
        seam_corners = []
        self.seam_masks = []

        seam_scale = min(1.0, np.sqrt(Config.smp / (Config.internal_panorama_width * Config.internal_panorama_width)))

        for idx, frame in enumerate(self.frames):
            img = cv.resize(frame.contents, dsize=None,
                            fx=seam_scale,
                            fy=seam_scale,
                            interpolation=cv.INTER_LINEAR_EXACT)
            seam_images.append(img.astype(np.float32))

            mask = cv.resize(frame.mask, dsize=None, fx=seam_scale,
                             fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
            self.seam_masks.append(mask)

            seam_corners.append((int(self.corners[idx][0] * seam_scale),
                                 int(self.corners[idx][1] * seam_scale)))

        umat_masks = seam_finder.find(seam_images, seam_corners, self.seam_masks)

        self.seam_masks = []
        for umat_mask in umat_masks:
            self.seam_masks.append(umat_mask.get())

    def blend_frames(self):
        self.blender.setNumBands((np.log(Config.frame_margin)/np.log(2.) - 1.).astype(np.int))

        dest_size = cv.detail.resultRoi(corners=self.corners,
                                        sizes=[(frame.width, frame.height) for frame in self.frames])
        self.blender.prepare(dest_size)

        for idx, frame in enumerate(self.frames):
            seam_mask = cv.dilate(self.seam_masks[idx], None)
            seam_mask = cv.resize(seam_mask, (frame.mask.shape[1], frame.mask.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask = cv.bitwise_and(seam_mask, frame.mask)

            self.blender.feed(cv.UMat(frame.contents), mask, self.corners[idx])

        result = None
        result_mask = None
        result, result_mask = self.blender.blend(result, result_mask)

        return result

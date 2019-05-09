import os

import numpy as np
import cv2 as cv
from libpano import Config
from libpano import utils
from libpano import ImageFrame


class Stitcher:

    def __init__(self, folder, meta, metrics):
        self.folder = folder
        self.meta = meta
        self.metrics = metrics

        self.blender = cv.detail_MultiBandBlender()
        self.seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR")
        self.images = []
        self.masks = []
        self.corners = []
        self.sizes = []

    def load_frames(self):
        cols = self.meta.col.nunique()

        for row in range(Config.start_row, Config.end_row):
            for col in range(cols):
                item = self.meta[(self.meta.row == row) & (self.meta.col == col)]
                fn = item.uri.values[0]
                fn = os.path.join(self.folder, fn)

                pitch = item.pitch.values[0]
                yaw = item.yaw.values[0]

                iframe = ImageFrame.ImageFrame(fn)
                iframe.load_image()

                corner = self.position_frame(pitch, yaw)

                self.images.append(iframe.contents)
                self.masks.append(iframe.mask)
                self.sizes.append((iframe.width, iframe.height))
                self.corners.append(corner)

    def position_frame(self, pitch, yaw):
        pitch = utils.degree2radian(pitch)
        yaw = utils.degree2radian(yaw)

        x = yaw * self.metrics.PPR_v + Config.frame_margin
        y = (pitch + np.pi / 2) * self.metrics.PPR_h + Config.frame_margin

        return int(x), int(y)

    def blend_frames(self):
        self.blender.setNumBands((np.log(Config.frame_margin)/np.log(2.) - 1.).astype(np.int))

        dest_size = cv.detail.resultRoi(corners=self.corners, sizes=self.sizes)
        self.blender.prepare(dest_size)

        for idx, img in enumerate(self.images):
            self.blender.feed(cv.UMat(img), self.masks[idx], self.corners[idx])

        result = None
        result_mask = None
        result, result_mask = self.blender.blend(result, result_mask)

        return result

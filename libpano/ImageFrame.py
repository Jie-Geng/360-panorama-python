import numpy as np
import cv2 as cv
import imutils
import os

from libpano import warpers
from libpano import Config


class ImageFrame:

    """
    Class ImageFrame represents a piece of mosaic image
    """

    def __init__(self, row, col, filename, pitch, yaw, roll):
        self.width = 0
        self.height = 0

        self.contents = None
        self.mask = None

        self.row = row
        self.col = col
        self.filename = filename
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    def load_image(self):
        self.contents = cv.imread(self.filename)

        self.height = self.contents.shape[0]
        self.width = self.contents.shape[1]

        # self.mask = np.zeros((self.height, self.width), np.uint8)
        # self.mask[:, :] = 255

    def preprocess_image(self, scale, metrics, temp_folder):
        self.load_image()

        # if warp first, warp frame
        if Config.order_warp_first:

            # warp it
            self.contents, self.mask = warpers.spherical_warp(self.contents, self.pitch, metrics)
            self.height = self.contents.shape[0]
            self.width = self.contents.shape[1]

            # crop too wide images
            if self.width > 1000:
                x = int((self.width - 1000) / 2)
                self.crop_rect(x, 0, 1000, self.height)
        else:
            # Create the mask only. Warping would be done in the positioning stage
            self.mask = np.zeros_like(self.contents, np.uint8)
            self.mask[:, :, :] = 255

        # resize
        height = int(self.height * scale)
        width = int(self.width * scale)

        self.contents = cv.resize(self.contents, (width, height), interpolation=cv.INTER_LINEAR_EXACT)
        self.mask = cv.resize(self.mask, (width, height), interpolation=cv.INTER_LINEAR_EXACT)
        self.mask = cv.cvtColor(self.mask, cv.COLOR_BGR2GRAY)

        # rotate
        self.contents = imutils.rotate_bound(self.contents, self.roll)
        self.mask = imutils.rotate_bound(self.mask, self.roll)

        self.height = self.contents.shape[0]
        self.width = self.contents.shape[1]

        # save
        file_name = 'warped-{}-{}.jpg'.format(self.row, self.col)
        cv.imwrite(os.path.join(temp_folder, file_name), self.contents)
        file_name = 'mask-{}-{}.jpg'.format(self.row, self.col)
        cv.imwrite(os.path.join(temp_folder, file_name), self.mask)

    def expand_size(self, new_width, new_height, center=True):
        big = np.zeros((new_height, new_width, 3), np.uint8)

        margin_x = 0
        margin_y = 0

        if center:
            margin_x = (new_width - self.width) // 2
            margin_y = (new_height - self.height) // 2

        big[margin_y:(margin_y + self.height), margin_x:(margin_x + self.width)] = self.contents

        del self.contents
        self.contents = big

    def crop_rect(self, x, y, width, height):
        small = self.contents[y:(y + height), x:(x + width)]
        self.contents = small

        small = self.mask[y:(y + height), x:(x + width)]
        self.mask = small

        self.width = width
        self.height = height

    def auto_crop(self):
        pass

    def save_to_file(self, filename):
        cv.imwrite(filename, self.contents)

    def free(self):
        del self.contents
        self.contents = None

import numpy as np
import cv2 as cv


class ImageFrame:

    """
    Class ImageFrame represents a piece of mosaic image
    """

    def __init__(self, filename):
        self.width = 0
        self.height = 0

        self.contents = None
        self.mask = None

        self.filename = filename

    def load_image(self):
        self.contents = cv.imread(self.filename)

        self.height = self.contents.shape[0]
        self.width = self.contents.shape[1]

        self.mask = np.zeros((self.height, self.width), np.uint8)
        self.mask[:, :] = 255

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

        del self.contents
        self.contents = small

    def auto_crop(self):
        pass

    def save_to_file(self, filename):
        cv.imwrite(filename, self.contents)

    def free(self):
        del self.contents
        self.contents = None

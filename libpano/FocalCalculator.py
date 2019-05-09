import os
import math
import numpy as np
import cv2 as cv


class FocalCalculator:

    meta = None
    metrics = None
    image_folder = None

    # OpenCV engines
    finder = None
    matcher = None
    estimator = None
    adjuster = None

    # buffers
    images = []
    image_names = []
    features = []
    matches = []

    # focal length
    focal = 0

    def __init__(self, image_folder, meta_data):
        self.image_folder = image_folder
        self.meta = meta_data.grid_data
        self.metrics = meta_data.metrics

        self.finder = cv.ORB_create()
        self.matcher = cv.detail.BestOf2NearestMatcher_create(False, 0.3)
        self.estimator = cv.detail_HomographyBasedEstimator()
        self.adjuster = cv.detail_BundleAdjusterRay()
        self.adjuster.setConfThresh(1)

    def load_and_compute_features(self, row):
        uris = self.meta[self.meta.row == row]['uri'].values.tolist()

        for uri in uris:
            self.image_names.append(uri)

            img = cv.imread(os.path.join(self.image_folder, uri))
            self.images.append(img)

            feature = cv.detail.computeImageFeatures2(self.finder, img)
            self.features.append(feature)

    def match(self):
        self.matches = self.matcher.apply2(self.features)
        self.matcher.collectGarbage()

        indices = cv.detail.leaveBiggestComponent(self.features, self.matches, 0.3)
        if len(indices) < 2:
            print('Cannot find matching images.')
            exit()

    def homography(self):
        success, cameras = self.estimator.apply(self.features, self.matches, None)

        if not success:
            print("Homography estimation failed.")
            exit()

        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        success, cameras = self.adjuster.apply(self.features, self.matches, cameras)
        if not success:
            print("Camera parameters adjusting failed.")
            exit()

        focals = []
        for cam in cameras:
            focals.append(cam.focal)

        sorted(focals)

        if len(focals) % 2 == 1:
            self.focal = focals[len(focals) // 2]
        else:
            self.focal = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

    @staticmethod
    def cylindrical_projection(img, focal_length):
        """
        This functions performs cylindrical warping, but its speed is slow and depricated.

        :param img: image contents
        :param focal_length:  focal length of images
        :return: warped image
        """
        height, width, _ = img.shape
        cylinder_proj = np.zeros(shape=img.shape, dtype=np.uint8)

        for y in range(-int(height / 2), int(height / 2)):
            for x in range(-int(width / 2), int(width / 2)):
                cylinder_x = focal_length * math.atan(x / focal_length)
                cylinder_y = focal_length * y / math.sqrt(x ** 2 + focal_length ** 2)

                cylinder_x = round(cylinder_x + width / 2)
                cylinder_y = round(cylinder_y + height / 2)

                if (cylinder_x >= 0) and (cylinder_x < width) and (cylinder_y >= 0) and (cylinder_y < height):
                    cylinder_proj[cylinder_y][cylinder_x] = img[y + int(height / 2)][x + int(width / 2)]

        # Crop black border
        # ref: http://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
        _, thresh = cv.threshold(cv.cvtColor(cylinder_proj, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv.boundingRect(contours[0])

        return cylinder_proj[y:y + h, x:x + w]

    @staticmethod
    def cylindrical_warp(img, k):
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        h_, w_ = img.shape[:2]

        # pixel coordinates
        y_i, x_i = np.indices((h_, w_))
        x = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homography
        k_inv = np.linalg.inv(k)
        x = k_inv.dot(x.T).T  # normalized coordinates

        # calculate cylindrical coordinates (sin\theta, h, cos\theta)
        a = np.stack([np.sin(x[:, 0]), x[:, 1], np.cos(x[:, 0])], axis=-1).reshape(w_ * h_, 3)
        b = k.dot(a.T).T  # project back to image-pixels plane
        # back from homography coordinates
        b = b[:, :-1] / b[:, [-1]]
        # make sure warp coordinates only within image bounds
        b[(b[:, 0] < 0) | (b[:, 0] >= w_) | (b[:, 1] < 0) | (b[:, 1] >= h_)] = -1
        b = b.reshape(h_, w_, -1)

        # img_rgba = cv.cvtColor(img, cv.COLOR_BGR2BGRA)  # for transparent borders...
        # warp the image according to cylindrical coordinates
        img = cv.remap(img,
                        b[:, :, 0].astype(np.float32),
                        b[:, :, 1].astype(np.float32),
                        cv.INTER_AREA,
                        borderMode=cv.BORDER_TRANSPARENT)

        # Crop black border
        # ref: http://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
        _, thresh = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv.boundingRect(contours[0])

        return img[y:y + h, x:x + w]

    def get_focal(self, row, do_cylindrical_warp=False):
        self.images = []
        self.image_names = []
        self.features = []
        self.matches = []

        self.load_and_compute_features(row)
        self.match()
        self.homography()

        print('{}th row\'s focal length = {}'.format(row, self.focal))

        if do_cylindrical_warp:
            for idx, image_name in enumerate(self.image_names):
                image = self.images[idx]

                h, w = image.shape[:2]
                k = np.array([[self.focal, 0, w / 2], [0, self.focal, h / 2], [0, 0, 1]])  # mock intrinsics
                image = self.cylindrical_warp(image, k)

                target_name = os.path.join(self.image_folder, image_name)
                print(target_name)
                cv.imwrite(target_name, image)

            print('Cylindrical projection was finished.')

        del self.images, self.image_names, self.features, self.matches

        return self.focal

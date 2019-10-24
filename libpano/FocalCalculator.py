import os
import numpy as np
import cv2 as cv

from libpano import warpers


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
    focals = []
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

        self.focals = focals

        focals = sorted(focals)

        if len(focals) % 2 == 1:
            self.focal = focals[len(focals) // 2]
        else:
            self.focal = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

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
                image = warpers.cylindrical_warp_with_k(image, k)

                target_name = os.path.join(self.image_folder, image_name)
                cv.imwrite(target_name, image)

        return self.focal

import os
import json

import pandas as pd
import numpy as np
import math
import cv2 as cv

from libpano import utils


class MetaData:
    """
    Class MetaData
    Manages the panorama meta data including grid data of image pieces and camera metrics
    """

    # pandas dataframe contains grid data of image pieces
    grid_data = None

    # PanoMetrics object contains metrics information
    metrics = None

    # paths
    json_path: str = None
    image_path: str = None

    # camera sensor data.
    sensor_width: float = 4.2336
    sensor_height: float = 5.6447997
    focal_length: float = 4.25

    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.image_path = base_folder

        self.get_meta_data()
        self.load_panorama_metrics(self.image_path)

    def get_meta_data(self):
        json_files = self.get_files_with_ext(self.base_folder, '.json')
        if len(json_files) == 0:
            raise Exception('argument', 'image folder does not contain any *.json file')

        self.json_path = json_files[0]

        # load json file and sort data into a pandas dataframe
        self.get_refined_dataframe()

    def load_panorama_metrics(self, image_path):
        self.image_path = image_path

        if self.grid_data is None:
            self.grid_data = self.get_meta_data()

        # calculate panorama metrics
        metrics = PanoMetrics(self.image_path,
                              self.grid_data,
                              self.sensor_width,
                              self.sensor_height,
                              self.focal_length,
                              from_overlap=False)
        self.metrics = metrics

    @staticmethod
    def get_files_with_ext(base_folder_path, ext, recursive=True):
        """
        Retrieves a list of all file paths with specific extension under the specified folder.

        :param base_folder_path: string. the folder in which files are searched
        :param ext: string. the extension to search for.
        :param recursive: boolean, default- True. if true searches subdirectories recursively.
        :return: list of strings.
        """

        dirs = [base_folder_path]
        found_files = list()

        while len(dirs) > 0:
            current_dir = dirs.pop(0)

            for file_name in os.listdir(current_dir):
                file_full_path = os.path.join(current_dir, file_name)

                if os.path.isfile(file_full_path):
                    if file_full_path.endswith(ext):
                        found_files.append(file_full_path)

                elif recursive:
                    dirs.append(file_full_path)

        return found_files

    def get_refined_dataframe(self):
        """
        Read JSON file and return the image file data frame.
        """

        # Read JSON file into a dataframe
        json_data = json.load(open(self.json_path, 'rt'))

        self.focal_length = json_data['focalLength']
        self.sensor_width = json_data['angleViewY']
        self.sensor_height = json_data['angleViewX']

        pictures = json_data['pictures']

        meta_pd = pd.DataFrame({
            'degree': [item['degrees'] for item in pictures],
            'ring': [item['ring'] for item in pictures],
            'uri': [item['sensors']['fileUri'] for item in pictures],
            'pitch': [item['sensors']['roll_pitch_yaw']['pitch'] for item in pictures],
            'roll': [item['sensors']['roll_pitch_yaw']['roll'] for item in pictures],
            'yaw': [item['sensors']['roll_pitch_yaw']['yaw'] for item in pictures]
        })

        # Refine image files sorted by pitch(vertical) and yaw(horizontal)
        df = pd.DataFrame({'row': [], 'col': [], 'uri': [], 'pitch': [], 'roll': [], 'yaw': []})

        sorted_rings = meta_pd.groupby(by='ring')['pitch'].agg(np.mean).sort_values().index

        sub_df = None
        for row, ring in enumerate(sorted_rings):
            sub_df = meta_pd[meta_pd.ring == ring].sort_values(by='yaw')
            sub_df['col'] = sub_df.reset_index().index
            sub_df['row'] = row

            df = df.append(sub_df[df.columns], ignore_index=False)

        df.row = df.row.astype('uint8')
        df.col = df.col.astype('uint8')
        df.reset_index(inplace=True)
        df.drop('index', axis=1, inplace=True)

        del json_data, meta_pd, sub_df

        self.grid_data = df


class PanoMetrics:
    """
    all angle members are in radian unit.
    all size members are in pixel unit.
    all intrinsic members are in mm unit.
    """

    # input data
    folder_path = None
    meta_data = None

    # Camera intrinsic measures
    sensor_v = 0
    sensor_h = 0
    focal_length = 0

    # frame counts
    N_v = 0
    N_h = 0

    # frame interval angle
    FIA_h = 0
    FIA_v = 0

    # frame size
    FW = 0
    FH = 0

    # panorama size
    PW = 0
    PH = 0

    # angle of view of the camera
    AOV_h = 0
    AOV_v = 0

    # angle of frame image
    AOF_h = 0
    AOF_v = 0

    # pixels per radian
    PPR_h = 0
    PPR_v = 0

    # sphere radius
    R_h = 0
    R_v = 0

    def __init__(self, folder_path,
                 meta_data,
                 sensor_width=0.0,
                 sensor_height=0.0,
                 focal_length=0.0,
                 from_overlap=True):
        """
        :param folder_path: path of the folder containing images
        :param meta_data: pandas dataframe containing image meta data
        :param sensor_width: width of camera sensor in millimeters
        :param sensor_height: height of camera sensor in millimeters
        :param focal_length: focal length of camera in millimeters
        :param from_overlap: True if we don't know the camera intrinsic features and have to
                    calculate them from image overlapping,
                    False if we know the camera intrinsic features
        """

        self.folder_path = folder_path
        self.meta_data = meta_data
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.focal_length = focal_length

        self.N_v = self.meta_data.row.nunique()
        self.N_h = self.meta_data.col.nunique()

        self.FIA_h = 2 * np.pi / self.N_h
        self.FIA_v = np.pi / self.N_v

        # get frame size from the center image
        uri = meta_data[(meta_data.row == self.N_v // 2) & (meta_data.col == self.N_h // 2)]['uri'].values[0]
        img = cv.imread(os.path.join(self.folder_path, uri))
        self.FH, self.FW, _ = img.shape
        del img

        # get angle of view
        if from_overlap:
            self.get_aov_h()
            self.get_aov_v()
        else:
            self.AOV_h = math.atan(self.sensor_width / (2 * self.focal_length)) * 2
            self.AOV_v = math.atan(self.sensor_height / (2 * self.focal_length)) * 2

        # get pixels per radian
        self.PPR_h = self.FW / self.AOV_h
        self.PPR_v = self.FH / self.AOV_v

        # get pixels per millimeter
        self.PPM_h = self.FW / self.sensor_width
        self.PPM_v = self.FH / self.sensor_height

        # focal length in pixels
        self.focal_length_px = self.PPM_h * self.focal_length

        # get radius of sphere
        self.R_h = self.PPR_h
        self.R_v = self.PPR_v

        # get panorama size
        self.PW = 2 * np.pi * self.R_h
        self.PH = np.pi * self.R_v

    def __str__(self):
        string = 'Camera Metrics:\n'
        string += '\tFocal Length: {} mm\n'.format(self.focal_length)
        string += '\tFocal Length: {:.2f} px\n'.format(self.focal_length_px)
        string += '\tSensor Size: {} x {} mm\n'.format(self.sensor_width, self.sensor_height)
        string += '\tPixels per mm: {:.2f} x {:.2f} px\n'.format(self.FW/self.sensor_width,
                                                                 self.FH/self.sensor_height)

        string += 'PanoramaMetrics:\n'
        string += '\tFrame Count: {} x {}\n'.format(self.N_h, self.N_v)
        string += '\tFrame Size: {}px x {}px\n'.format(self.FW, self.FH)
        string += '\tInterval Angle: {}︒ x {}︒\n'.format(utils.radian2degree(self.FIA_h),
                                                         utils.radian2degree(self.FIA_v))
        string += '\tAoV: {:.4f}︒ x {:.4f}︒\n'.format(utils.radian2degree(self.AOV_h),
                                                      utils.radian2degree(self.AOV_v))
        string += '\tAoV: {:.4f} x {:.4f}\n'.format(self.AOV_h, self.AOV_v)
        string += '\tPPR: {:.4f}px x {:.4f}px\n'.format(self.PPR_h, self.PPR_v)
        string += '\tPanorama Size: {:.4f}px x {:.4f}px\n'.format(self.PW, self.PH)
        return string

    @staticmethod
    def get_non_overlapping_width(image_file1, image_file2):
        """
        Get the average NOW(non overlapping width) of image frame from two adjacent frames

        |<-- NOW --->|

        -------------|------|------------
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        |            |      |           |
        -------------|------|------------

        |---- Image 1 ------|

                     |------ Image2 ----|


        """

        # Load the images
        img1 = cv.imread(image_file1)
        img2 = cv.imread(image_file2)

        # Detect the ORB key points and compute the descriptors for the two images
        orb = cv.ORB_create()
        key_points1, descriptors1 = orb.detectAndCompute(img1, None)
        key_points2, descriptors2 = orb.detectAndCompute(img2, None)

        # Create brute-force matcher object
        bf = cv.BFMatcher()

        # Match the descriptors
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Select the good matches using the ratio test
        good_matches = []

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        del matches
        matches = good_matches

        # Apply the homography transformation if we have enough good matches
        min_match_count = 10

        if len(good_matches) < min_match_count:
            return -1

        # Get the good key points positions
        source_points = np.float32([key_points1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        destination_points = np.float32([key_points1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Obtain the homography matrix
        _, mask = cv.findHomography(source_points, destination_points, method=cv.RANSAC, ransacReprojThreshold=5.0)
        matches_mask = mask.ravel().tolist()

        # filter masked matches
        filtered_matches = []
        for idx, m in enumerate(matches):
            if matches_mask[idx] == 1:
                filtered_matches.append(m)

        del matches
        matches = filtered_matches

        # get average NOW
        non_overlapping_widths = [key_points1[m.queryIdx].pt[0] - key_points2[m.trainIdx].pt[0] for m in matches]

        del img1, img2
        return np.mean(non_overlapping_widths)

    @staticmethod
    def get_non_overlapping_height(image_file1, image_file2):
        """
        Get the average NOH(non overlapping height) of image frame from two vertically adjacent frames

          -  ---------------------  ---
          |  |                   |    |
          |  |                   |    |
          |  |                   |    |
         NOH |                   |    |
          |  |                   |    |
          |  |                   | Image 1
          |  |                   |    |
          -  ---------------------  --+---
             |                   |    |  |
             |    Overlapping    |    |  |
             |                   |    |  |
             ---------------------  ---  |
             |                   |       |
             |                   |       |
             |                   |    Image 2
             |                   |       |
             |                   |       |
             |                   |       |
             |                   |       |
             ---------------------  ------
        """

        # Load the images
        img1 = cv.imread(image_file1)
        img2 = cv.imread(image_file2)

        # Detect the ORB key points and compute the descriptors for the two images
        orb = cv.ORB_create()
        key_points1, descriptors1 = orb.detectAndCompute(img1, None)
        key_points2, descriptors2 = orb.detectAndCompute(img2, None)

        # Create brute-force matcher object
        bf = cv.BFMatcher()

        # Match the descriptors
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Select the good matches using the ratio test
        good_matches = []

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        del matches
        matches = good_matches

        # Apply the homography transformation if we have enough good matches
        min_match_count = 10

        if len(good_matches) < min_match_count:
            return -1

        # Get the good key points positions
        source_points = np.float32([key_points1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        destination_points = np.float32([key_points2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Obtain the homography matrix
        _, mask = cv.findHomography(source_points, destination_points, method=cv.RANSAC, ransacReprojThreshold=5.0)
        matches_mask = mask.ravel().tolist()

        # filter masked matches
        filtered_matches = []
        for idx, m in enumerate(matches):
            if matches_mask[idx] == 1:
                filtered_matches.append(m)

        del matches
        matches = filtered_matches

        # get NOH
        non_overlapping_widths = [key_points1[m.queryIdx].pt[1] - key_points2[m.trainIdx].pt[1] for m in matches]

        del img1, img2

        return np.mean(non_overlapping_widths)

    def get_aov_h(self):
        # the most horizontal row
        best_row = self.meta_data['row'].max() // 2

        # yaw differences in the middle row
        yaws = self.meta_data[self.meta_data.row == best_row]['yaw'].tolist()
        yaw_interval = self.FIA_h
        yaw_diffs = [abs(yaws[i + 1] - yaws[i]) for i in range(len(yaws) - 1)]
        yaw_diffs = [abs(diff - yaw_interval) for diff in yaw_diffs]

        # get the index whose yaw diff is minimum
        best_col = np.argmin(yaw_diffs)

        # get the two adjacent image names from the best col
        fn1 = self.meta_data[(self.meta_data.row == best_row) & (self.meta_data.col == best_col)]['uri'].values[0]
        fn2 = self.meta_data[(self.meta_data.row == best_row) & (self.meta_data.col == best_col + 1)]['uri'].values[0]

        fn1 = self.folder_path + fn1
        fn2 = self.folder_path + fn2

        # get non overlapping width
        now = self.get_non_overlapping_width(fn1, fn2)

        # calculate angle of view(horizontal)
        self.AOV_h = self.FIA_h * self.FW / now

    def get_aov_v(self):
        # the most horizontal row
        best_row = self.meta_data['row'].max() // 2

        # pitch differences in the middle row and the next row
        pitches = self.meta_data[self.meta_data.row == best_row]['pitch'].tolist()
        pitches1 = self.meta_data[self.meta_data.row == best_row + 1]['pitch'].tolist()

        pitch_interval = self.FIA_v
        pitch_diffs = [abs(pitches1[i] - pitches[i]) for i in range(len(pitches))]
        pitch_diffs = [abs(diff - pitch_interval) for diff in pitch_diffs]

        # get the index whose pitch diff is minimum
        best_col = np.argmin(pitch_diffs)

        # get the two vertically adjacent image names from the best col
        fn1 = self.meta_data[(self.meta_data.row == best_row) & (self.meta_data.col == best_col)]['uri'].values[0]
        fn2 = self.meta_data[(self.meta_data.row == best_row + 1) & (self.meta_data.col == best_col)]['uri'].values[0]

        fn1 = self.folder_path + fn1
        fn2 = self.folder_path + fn2

        # get non overlapping height
        noh = self.get_non_overlapping_height(fn1, fn2)

        # calculate angle of view(vertical)
        self.AOV_v = self.FIA_v * self.FH / noh

import os

import numpy as np
import pandas as pd
import cv2 as cv
from multiprocessing import Pool

from libpano import Config
from libpano import utils
from libpano import ImageFrame


def preprocess_frame(args):
    frame, scale, metrics, temp_folder = args
    frame.preprocess_image(scale, metrics, temp_folder)

    return frame


class Stitcher:

    def __init__(self, folder, temp_folder, meta):
        # input attributes
        self.folder = folder
        self.temp_folder = temp_folder
        self.meta = meta.grid_data
        self.metrics = meta.metrics

        self.frames = []

        # stitcher components
        self.blender = cv.detail_MultiBandBlender()
        self.compensator = cv.detail_ChannelsCompensator(2)

        # seam attributes
        self.corners = []
        self.sizes = []

        self.seam_masks = []

        # positioning attributes
        self.grid_data = None
        self.ppd_x = 0
        self.ppd_y = 0
        self.other_rows = []

    def load_and_preprocess(self, scale=0, rows=None):

        n_rows = self.meta.row.nunique()
        n_cols = self.meta.col.nunique()

        process_rows = rows
        if process_rows is None:
            process_rows = [r for r in range(n_rows)]

        # process_rows = [1]
        # n_cols = 3
        for row in process_rows:
            for col in range(n_cols):
                item = self.meta[(self.meta.row == row) & (self.meta.col == col)]
                fn = item.uri.values[0]
                fn = os.path.join(self.folder, fn)

                pitch = item.pitch.values[0]
                yaw = item.yaw.values[0]
                roll = item.roll.values[0]

                frame = ImageFrame.ImageFrame(row, col, fn,
                                              utils.degree2radian(pitch),
                                              utils.degree2radian(yaw),
                                              roll)
                self.frames.append(frame)

        if scale == 0:
            # calculate the resize scale
            scale = Config.internal_panorama_width / self.metrics.PW

        use_mp = False

        if use_mp:
            # Do it on every core.
            pool = Pool(processes=None)

            # prepare arg
            args = []
            for frame in self.frames:
                args.append((frame, scale, self.metrics, self.temp_folder))

            frames = pool.map(preprocess_frame, args)
            self.frames = frames
        else:
            for frame in self.frames:
                print(frame.filename)
                frame.preprocess_image(scale, self.metrics, self.temp_folder)

    @staticmethod
    def position_get_real_width(mask):
        # get contours
        gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        _, threshold = cv.threshold(gray, 5, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # choose the biggest contour
        max_area = 0
        best_cnt = 0
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = cnt

        # get the real width
        x, y, w, h = cv.boundingRect(best_cnt)

        return w

    def position_adjust_extreme(self):
        extreme = self.grid_data[(self.grid_data.width > 2000)]
        for idx, record in extreme.iterrows():
            row = record['row']
            col = record['col']

            x = record['x']
            img = cv.imread('{}/mask-{}-{}.jpg'.format(self.temp_folder, row, col))

            total_width = record['width']
            left_width = self.position_get_real_width(img[:, :int(total_width / 2)])
            right_width = self.position_get_real_width(img[:, int(total_width / 2):])

            real_width = left_width + right_width

            if left_width < right_width:
                x += total_width - right_width
            else:
                x -= right_width

            self.grid_data.at[idx, 'width'] = real_width
            self.grid_data.at[idx, 'x'] = x

            # refine mask
            new_img = np.zeros((img.shape[0], real_width, 3), np.uint8)
            new_img[:, right_width:] = img[:, :left_width]
            new_img[:, :right_width] = img[:, -right_width:]
            cv.imwrite('{}/mask-{}-{}.jpg'.format(self.temp_folder, row, col), new_img)

            # refine image
            img = cv.imread('{}/warped-{}-{}.jpg'.format(self.temp_folder, row, col))
            new_img = np.zeros((img.shape[0], real_width, 3), np.uint8)
            new_img[:, right_width:] = img[:, :left_width]
            new_img[:, :right_width] = img[:, -right_width:]
            cv.imwrite('{}/warped-{}-{}.jpg'.format(self.temp_folder, row, col), new_img)

    def position_update_ppd(self, target_rows):
        pitch_and_cy = self.grid_data[self.grid_data.x.notnull()].groupby(by='row')['pitch', 'cy'].agg(np.mean)
        self.ppd_y = np.mean(np.diff(pitch_and_cy['cy'].values) / np.diff(pitch_and_cy['pitch'].values))

        ppd_cx = 0
        for row in range(self.metrics.N_v):
            if row in target_rows:
                continue
            yaw_and_cx = self.grid_data[self.grid_data.row == row][['yaw', 'cx']]
            ppd_cx += np.mean(np.diff(yaw_and_cx['cx'].values) / np.diff(yaw_and_cx['yaw'].values))
        self.ppd_x = ppd_cx / (self.metrics.N_v - len(target_rows))

    def position_infer_coordinates(self, target_rows):
        c_row = self.metrics.N_v // 2

        for i in range(c_row+1):

            # northern hemisphere rows
            row = c_row - i

            if row in target_rows:
                print('processing', str(row))
                for col in range(self.metrics.N_h):
                    # infer cy value
                    below_cy = self.grid_data[(self.grid_data.row == row + 1) &
                                              (self.grid_data.col == col)].cy.values[0]
                    below_pitch = self.grid_data[(self.grid_data.row == row + 1) &
                                                 (self.grid_data.col == col)].pitch.values[0]
                    my_pitch = self.grid_data[(self.grid_data.row == row) &
                                              (self.grid_data.col == col)].pitch.values[0]

                    my_cy = below_cy - (below_pitch - my_pitch) * self.ppd_y
                    self.grid_data.at[(self.grid_data.row == row) & (self.grid_data.col == col), 'cy'] = my_cy

                    # calculate y
                    my_height = self.grid_data[(self.grid_data.row == row) &
                                               (self.grid_data.col == col)].height.values[0]
                    self.grid_data.at[(self.grid_data.row == row) &
                                      (self.grid_data.col == col), 'y'] = my_cy - my_height/2

                    # infer cx value
                    below_cx = self.grid_data[(self.grid_data.row == row + 1) &
                                              (self.grid_data.col == col)].cx.values[0]
                    below_yaw = self.grid_data[(self.grid_data.row == row + 1) &
                                               (self.grid_data.col == col)].yaw.values[0]
                    my_yaw = self.grid_data[(self.grid_data.row == row) &
                                            (self.grid_data.col == col)].yaw.values[0]

                    my_cx = below_cx + (my_yaw - below_yaw) * self.ppd_x
                    self.grid_data.at[(self.grid_data.row == row) & (self.grid_data.col == col), 'cx'] = my_cx

                    # calculate x
                    my_width = self.grid_data[(self.grid_data.row == row) &
                                              (self.grid_data.col == col)].width.values[0]
                    self.grid_data.at[(self.grid_data.row == row) &
                                      (self.grid_data.col == col), 'x'] = my_cx - my_width / 2

            # southern hemisphere rows
            row = c_row + i
            if row not in target_rows:
                continue
            if row >= self.metrics.N_h:
                continue

            print('processing', str(row))
            for col in range(self.metrics.N_h):
                # infer cy value
                above_cy = self.grid_data[(self.grid_data.row == row - 1) & (self.grid_data.col == col)].cy.values[0]
                above_pitch = self.grid_data[(self.grid_data.row == row - 1) &
                                             (self.grid_data.col == col)].pitch.values[0]
                my_pitch = self.grid_data[(self.grid_data.row == row) &
                                          (self.grid_data.col == col)].pitch.values[0]

                my_cy = above_cy + (my_pitch - above_pitch) * self.ppd_y
                self.grid_data.at[(self.grid_data.row == row) & (self.grid_data.col == col), 'cy'] = my_cy

                # infer y
                my_height = self.grid_data[(self.grid_data.row == row) & (self.grid_data.col == col)].height.values[0]
                self.grid_data.at[(self.grid_data.row == row) & (self.grid_data.col == col), 'y'] = my_cy - my_height/2

                # infer cx value
                above_cx = self.grid_data[(self.grid_data.row == row - 1) & (self.grid_data.col == col)].cx.values[0]
                above_yaw = self.grid_data[(self.grid_data.row == row - 1) & (self.grid_data.col == col)].yaw.values[0]
                my_yaw = self.grid_data[(self.grid_data.row == row) & (self.grid_data.col == col)].yaw.values[0]

                my_cx = above_cx + (my_yaw - above_yaw) * self.ppd_x
                self.grid_data.at[(self.grid_data.row == row) & (self.grid_data.col == col), 'cx'] = my_cx

                # calculate x
                my_width = self.grid_data[(self.grid_data.row == row) & (self.grid_data.col == col)].width.values[0]
                self.grid_data.at[(self.grid_data.row == row) & (self.grid_data.col == col), 'x'] = my_cx - my_width / 2

        self.grid_data.to_csv('/tmp/middle.csv')
        self.grid_data.x.fillna(0, inplace=True)
        self.grid_data.y.fillna(0, inplace=True)
        self.grid_data.width.fillna(0, inplace=True)
        self.grid_data.height.fillna(0, inplace=True)
        self.grid_data.x = self.grid_data.x.astype('int32')
        self.grid_data.y = self.grid_data.y.astype('int32')
        self.grid_data.width = self.grid_data.width.astype('int32')
        self.grid_data.height = self.grid_data.height.astype('int32')

    def position_frames(self, target_rows):
        self.other_rows = target_rows

        # read prestitch result data
        frame_file_name = os.path.join(self.temp_folder, Config.register_result_name)
        ps_result = pd.read_csv(frame_file_name, delimiter=' ', header=None,
                                names=['row', 'col', 'x', 'y', 'width', 'height'])

        self.grid_data = pd.merge(self.meta, ps_result, on=['row', 'col'], how='outer')

        # fill the size data of the pre-positioned frames
        # for frame in self.frames:
        #     if frame.row not in target_rows:  # useless as frames are in only non-prestitch rows.
        #         continue
        #
        #     self.grid_data.at[(self.grid_data.row == frame.row) &
        #                       (self.grid_data.col == frame.col), 'width'] = frame.width
        #     self.grid_data.at[(self.grid_data.row == frame.row) &
        #                       (self.grid_data.col == frame.col), 'height'] = frame.height
        for row in target_rows:
            for col in range(self.metrics.N_h):
                img = cv.imread('{}/warped-{}-{}.jpg'.format(self.temp_folder, row, col))
                self.grid_data.at[(self.grid_data.row == row) &
                                  (self.grid_data.col == col), 'width'] = img.shape[1]
                self.grid_data.at[(self.grid_data.row == row) &
                                  (self.grid_data.col == col), 'height'] = img.shape[0]
                del img


        # adjust extreme frames
        # self.position_adjust_extreme()

        # create center coordinate columns
        self.grid_data = self.grid_data.assign(cx=self.grid_data.x + self.grid_data.width / 2,
                                               cy=self.grid_data.y + self.grid_data.height / 2)

        # adjust yaws according to the cv prestitcher result
        self.grid_data = self.grid_data.assign(yaw=[yaw - 360 if yaw > 180 else yaw for yaw in self.grid_data.yaw])

        # get ppd_x(pixels per degree in x axis) and ppd_y(pixels per degree in y axis)
        self.position_update_ppd(target_rows)

        # infer cx, x, cy, and y
        self.position_infer_coordinates(target_rows)

        file_name = os.path.join(self.temp_folder, 'grid-data.csv')
        self.grid_data.to_csv(file_name, sep=' ', index=False)

        if Config.mainframe_first:
            other_df = self.grid_data[self.grid_data.row.isin(target_rows)]

            rows = other_df.row.values.tolist()
            cols = other_df.col.values.tolist()
            image_names = ['warped-{}-{}.jpg'.format(row, col) for row, col in zip(rows, cols)]
            mask_names = ['mask-{}-{}.jpg'.format(row, col) for row, col in zip(rows, cols)]
            x = other_df.x.values.tolist()
            y = other_df.y.values.tolist()

            frame_x = ps_result.x.min()
            frame_y = ps_result.y.min()
            image_names.insert(0, 'frame.jpg')
            mask_names.insert(0, 'frame-mask.jpg')
            x.insert(0, frame_x)
            y.insert(0, frame_y)

            compose_df = pd.DataFrame({'img_name': image_names, 'mask_name': mask_names, 'x': x, 'y': y})

            file_name = os.path.join(self.temp_folder, Config.compose_config_name)
            compose_df.to_csv(file_name, sep=' ', header=False, index=False)

    def seam_find(self):
        seam_finder = cv.detail_DpSeamFinder("COLOR_GRAD")

        seam_images = []
        seam_corners = []
        self.seam_masks = []

        seam_scale = min(1.0, np.sqrt(Config.smp / (Config.internal_panorama_width * Config.internal_panorama_width)))

        # for idx, row in self.grid_data.iterrows():
        #     if Config.only_main_frame:
        #         if row.row in self.other_rows:
        #             continue
        #     # add warped images
        #     file_name = os.path.join(self.temp_folder, 'warped-{}-{}.jpg'.format(row.row, row.col))
        #     big_image = cv.imread(file_name)
        #     small_img = cv.resize(big_image, dsize=None,
        #                           fx=seam_scale,
        #                           fy=seam_scale,
        #                           interpolation=cv.INTER_LINEAR_EXACT)
        #     seam_images.append(small_img.astype(np.float32))
        #
        #     # add warped masks
        #     file_name = os.path.join(self.temp_folder, 'mask-{}-{}.jpg'.format(row.row, row.col))
        #     big_image = cv.imread(file_name)
        #     big_image = cv.cvtColor(big_image, cv.COLOR_BGR2GRAY)
        #     small_img = cv.resize(big_image, dsize=None,
        #                           fx=seam_scale,
        #                           fy=seam_scale,
        #                           interpolation=cv.INTER_LINEAR_EXACT)
        #     self.seam_masks.append(small_img.astype(np.uint8))
        #
        #     seam_corners.append((int(row.x * seam_scale), int(row.y * seam_scale)))
        df = pd.read_csv(os.path.join(self.temp_folder, Config.compose_config_name), header=None, delimiter=' ',
                         names=['image', 'mask', 'x', 'y'])
        for _, row in df.iterrows():
            file_name = os.path.join(self.temp_folder, row['image'])
            big_image = cv.imread(file_name)
            small_img = cv.resize(big_image, dsize=None, fx=seam_scale, fy=seam_scale,
                                  interpolation=cv.INTER_LINEAR_EXACT)
            seam_images.append(small_img)

            file_name = os.path.join(self.temp_folder, row['mask'])
            big_image = cv.imread(file_name)
            big_image = cv.cvtColor(big_image, cv.COLOR_BGR2GRAY)
            small_img = cv.resize(big_image, dsize=None, fx=seam_scale, fy=seam_scale,
                                  interpolation=cv.INTER_LINEAR_EXACT)
            self.seam_masks.append(small_img)

            seam_corners.append((int(row.x * seam_scale), int(row.y * seam_scale)))
            self.corners.append((int(row.x), int(row.y)))
            self.sizes.append((big_image.shape[1], big_image.shape[0]))

        # compensator feed
        self.compensator.feed(corners=seam_corners, images=seam_images, masks=self.seam_masks)

        # find seam
        umat_masks = seam_finder.find(seam_images, seam_corners, self.seam_masks)

        self.seam_masks = []
        for umat_mask in umat_masks:
            self.seam_masks.append(umat_mask.get())

        self.grid_data = df

    def blend_frames(self):

        # get corners and sizes list
        # corners = []
        # sizes = []
        # for _, row in self.grid_data.iterrows():
        #     if Config.only_main_frame:
        #         if row.row in self.other_rows:
        #             continue
        #
        #     sizes.append((int(row.width), int(row.height)))
        #     corners.append((int(row.x), int(row.y)))

        dest_size = cv.detail.resultRoi(corners=self.corners, sizes=self.sizes)
        blend_width = np.sqrt(dest_size[2] * dest_size[3]) / 50
        self.blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))

        self.blender.prepare(dest_size)

        idx = 0
        for _, row in self.grid_data.iterrows():
            if Config.only_main_frame:
                if row.row in self.other_rows:
                    continue

            # read and apply seaming result
            # file_name = os.path.join(self.temp_folder, 'mask-{}-{}.jpg'.format(row.row, row.col))
            file_name = os.path.join(self.temp_folder, row['mask'])
            mask = cv.imread(file_name)
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

            seam_mask = cv.dilate(self.seam_masks[idx], None)
            seam_mask = cv.dilate(seam_mask, None)
            seam_mask = cv.resize(seam_mask, (mask.shape[1], mask.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            seam_mask = cv.GaussianBlur(seam_mask, (25, 25), 0, 0)
            _, seam_mask = cv.threshold(seam_mask, 127, 255, cv.THRESH_BINARY)
            seam_mask = seam_mask.astype(np.uint8)

            mask = cv.bitwise_and(seam_mask, mask)

            # load image
            # file_name = os.path.join(self.temp_folder, 'warped-{}-{}.jpg'.format(row.row, row.col))
            file_name = os.path.join(self.temp_folder, row['image'])
            image = cv.imread(file_name)

            image = self.compensator.apply(idx, self.corners[idx], image, mask)
            self.blender.feed(cv.UMat(image), mask, self.corners[idx])

            idx += 1

        result = None
        result_mask = None
        result, result_mask = self.blender.blend(result, result_mask)

        return result

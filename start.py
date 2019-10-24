import argparse
import os
import shutil
import subprocess
import tempfile
import cv2 as cv
import math
import pandas as pd
import numpy as np

from libpano import MetaData
from libpano import Stitcher
from libpano import utils
from libpano import Config
from libpano import warpers

parser = argparse.ArgumentParser(prog='start.py', description='my panorama stitch program')
parser.add_argument('folder', nargs='+', help='folder containing files to stitch', type=str)
parser.add_argument('--width', help='Width of the output panorama', type=int, dest='width')
parser.add_argument('--height', help='Height of the output panorama', type=int, dest='height')
parser.add_argument('--output', default='panorama.jpg', help='File name of the output file.', type=str,
                    dest='output')

__doc__ = '\n' + parser.format_help()


def main():
    all_timer = utils.Timer()

    args = parser.parse_args()
    base_folder = args.folder[0]
    pano_width = args.width
    pano_height = args.height
    output_fn = args.output

    ############################################
    # load meta data
    ############################################

    meta = MetaData.MetaData(base_folder)
    print('')
    print(meta.metrics)

    # non-processed rows
    if meta.metrics.N_v == 4:
        other_rows = [0, 3]
    elif meta.metrics.N_v == 5:
        # other_rows = [0, 1, 4]
        other_rows = [0, 4]
    elif meta.metrics.N_v == 6:
        other_rows = [0, 1, 4, 5]
    elif meta.metrics.N_v == 7:
        other_rows = [0, 1, 5, 6]
    elif meta.metrics.N_v == 8:
        other_rows = [0, 1, 6, 7]
    elif meta.metrics.N_v == 9:
        other_rows = [0, 1, 2, 7, 8]
    else:
        print("\nERROR: number of rows should be in 4 ~ 9.\n")
        return -1

    ############################################
    # Pre-stitching of mainframe rows
    ############################################

    # make temporary work directory
    temp_folder = tempfile.mkdtemp()

    # write meta data for the prestitcher
    meta_string = meta.meta_to_string()
    meta_file_name = os.path.join(temp_folder, Config.meta_data_name)
    f = open(meta_file_name, "w")
    f.write(meta_string)
    f.close()

    # calculate image scale
    scale = math.sqrt(Config.internal_panorama_width / meta.metrics.PW)
    # TODO: fix the scale value just now
    # if the scale value is double value, the register runs difficultly.
    scale = 0.5
    print('scale={}'.format(scale))

    ############################################
    # Stitch the mainframe panorama
    ############################################
    # print("- Registering....")
    # command = ['utils/pano-register',
    #            '--folder', base_folder,
    #            '--temp-folder', temp_folder,
    #            '--meta', meta_file_name,
    #            '--scale', str(scale),
    #            '--verbose']
    # return_code = subprocess.call(command)
    #
    # if return_code != 0:
    #     print("Error in registering frame images.")
    #     return -1
    #
    # if Config.mainframe_first:
    #     print("- Composing.....")
    #     return_code = subprocess.call(['utils/pano-composer',
    #                                    '--folder', temp_folder,
    #                                    '--config', Config.register_result_name,
    #                                    '--mode', 'frame',
    #                                    '--output', os.path.join(temp_folder, 'frame.jpg')])
    #
    #     if return_code != 0:
    #         print("Error in composing frame images.")
    #         return -1
    
    ############################################
    # Transforming other rows(not-mainframe rows)
    ############################################

    # Calculate the scale done by prestitcher
    # psr_name = os.path.join(temp_folder, Config.register_result_name)
    # psr_df = pd.read_csv(psr_name, header=0, delimiter=' ', names=['row', 'col' 'x', 'y', 'width', 'height'])

    temp_folder = '/home/jie/work/pano/201708/10'
    # resizing, warping, and rotating
    stitcher = Stitcher.Stitcher(base_folder, temp_folder, meta)

    timer = utils.Timer()
    print('\n- Load and preprocess images.....', end='', flush=True)
    # stitcher.load_and_preprocess(scale, other_rows)
    stitcher.load_and_preprocess(1.0, other_rows)
    print('{:.3f} seconds'.format(timer.end()))
    return 0

    print('- Positioning images.....', end='')
    timer.begin()
    stitcher.position_frames(other_rows)
    print('{:.3f} seconds'.format(timer.end()))

    if Config.mainframe_first:
        print("- Composing.....")
        raw_output_name = os.path.join(temp_folder, 'panorama.jpg')
        return_code = subprocess.call(['utils/pano-composer',
                                       '--folder', temp_folder,
                                       '--config', Config.compose_config_name,
                                       '--mode', 'full',
                                       '--output', raw_output_name])
 
        if return_code != 0:
            print("Error in composing frame images.")
            return -1

        output = cv.imread(raw_output_name)

    else:
        # Now, we don't need frames any more
        # stitcher.grid_data = pd.read_csv(os.path.join(temp_folder, Config.compose_config_name),
        #                                  sep=' ', header=[''])

        stitcher.frames = []

        # seam finding
        print('- Finding seams.....', end='', flush=True)
        timer.begin()
        stitcher.seam_find()
        print('{:.3f} seconds'.format(timer.end()))

        # blending images
        print('- Blending images.....', end='', flush=True)
        timer.begin()
        output = stitcher.blend_frames()
        print('{:.3f} seconds'.format(timer.end()))

    cv.imwrite('/home/gengjie/temp/pano.jpg', output)
    ############################################
    # Cropping, saving, removing temporary folder
    ############################################
    print('- Cropping and resizing.....', end='')
    timer.begin()

    if False:
        height, width = output.shape[0], output.shape[1]
        cy = height // 2

        gray = cv.cvtColor(output.astype(np.uint8), cv.COLOR_BGR2GRAY)
        middle_line = gray[cy, :]

        left = 0
        while middle_line[left] == 0:
            left += 1

        right = width - 1
        while middle_line[right] == 0:
            right -= 1

        roi_width = right - left
        roi_height = roi_width // 2 - 200
        top = (height - roi_height) // 2

        cropped = output[top:(top + roi_height), left:(left + roi_width), :]
        output = cv.resize(cropped, (pano_width, pano_height), 0, 0, cv.INTER_LINEAR_EXACT)

    cv.imwrite(output_fn, output)

    # try:
    #     shutil.rmtree(temp_folder)
    # except OSError:
    #     pass

    print('{:.3f} seconds'.format(timer.end()))

    print('\nPanorama was created and stored as {} in {:.2f} seconds'.format(output_fn, all_timer.end()))

    return 0


if __name__ == '__main__':
    main()

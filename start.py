import argparse
import os
import shutil
import subprocess
import tempfile
import cv2 as cv
import math
import pandas as pd
from libpano import MetaData
from libpano import Stitcher
from libpano import utils
from libpano import Config

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
    print("CAMERA METRICS:")
    print(meta.metrics)

    # non-processed rows
    if meta.metrics.N_v == 4:
        other_rows = [0, 3]
    elif meta.metrics.N_v == 5:
        other_rows = [0, 1, 4]
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
    print('scale={}'.format(scale))

    print("\n\nPrestitching....")
    return_code = subprocess.call(['utils/prestitcher',
                                   '--folder', base_folder,
                                   '--temp-folder', temp_folder,
                                   '--meta', meta_file_name,
                                   '--scale', str(scale)])

    if return_code != 0:
        print("Error in prestitching.")
        return -1

    ############################################
    # Transforming other rows(not-mainframe rows)
    ############################################

    # Calculate the scale done by prestitcher
    psr_name = os.path.join(temp_folder, Config.prestitch_result_name)
    psr_df = pd.read_csv(psr_name, header=0, names=['row', 'col' 'x', 'y', 'width', 'height'])
    minimum_width = psr_df['width'].min()
    other_scale = minimum_width / meta.metrics.FW
    print('min_width={}, scale={}'.format(minimum_width, other_scale))

    # resizing, warping, and rotating
    stitcher = Stitcher.Stitcher(base_folder, temp_folder, meta)

    timer = utils.Timer()
    print('- Load and preprocess images.....', end='', flush=True)
    stitcher.load_and_preprocess(1.0, other_rows)
    print('{:.3f} seconds'.format(timer.end()))

    print('- Positioning images.....', end='')
    timer.begin()
    stitcher.position_frames(other_rows)
    print('{:.3f} seconds'.format(timer.end()))

    # Now, we don't need frames any more
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

    cv.imwrite(output_fn, output)

    print('\nDone in {:.2f} seconds'.format(all_timer.end()))

    return 0


if __name__ == '__main__':
    main()

import argparse
import os
import shutil
import cv2 as cv
from libpano import MetaData
from libpano import Preprocess
from libpano import Stitcher
from libpano import Config
from libpano import utils
from libpano import FocalCalculator

from libpano import manual

parser = argparse.ArgumentParser(prog='start.py', description='my panorama stitch program')
parser.add_argument('folder', nargs='+', help='folder containing files to stitch', type=str)
parser.add_argument('--width', help='Width of the output panorama', type=int, dest='width')
parser.add_argument('--height', help='Height of the output panorama', type=int, dest='height')
parser.add_argument('--output', default='output.jpg', help='File name of the output file.', type=str,
                    dest='output')

__doc__ = '\n' + parser.format_help()


def main():
    all_timer = utils.Timer()
    timer = utils.Timer()

    args = parser.parse_args()
    base_folder = args.folder[0]
    pano_width = args.width
    pano_height = args.height
    output_fn = args.output

    meta = MetaData.MetaData(base_folder)

    stitcher = Stitcher.Stitcher(base_folder, meta)
    print('- Load and preprocess images..')
    stitcher.load_and_preprocess()
    print('- Positioning images..')
    stitcher.position_frames()

    if Config.debug:
        print('Finished in {:.2f} seconds'.format(timer.end()))

    # blending images
    if Config.debug:
        timer.begin()
        print('Blending images...')

    output = stitcher.blend_frames()
    cv.imwrite(output_fn, output)

    if Config.debug:
        print('Finished in {:.2f} seconds'.format(timer.end()))

    print('\nDone in {:.2f} seconds'.format(all_timer.end()))


if __name__ == '__main__':
    main()

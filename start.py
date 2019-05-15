import argparse
import cv2 as cv
from libpano import MetaData
from libpano import Stitcher
from libpano import utils

parser = argparse.ArgumentParser(prog='start.py', description='my panorama stitch program')
parser.add_argument('folder', nargs='+', help='folder containing files to stitch', type=str)
parser.add_argument('--width', help='Width of the output panorama', type=int, dest='width')
parser.add_argument('--height', help='Height of the output panorama', type=int, dest='height')
parser.add_argument('--output', default='output.jpg', help='File name of the output file.', type=str,
                    dest='output')

__doc__ = '\n' + parser.format_help()


def main():
    all_timer = utils.Timer()

    args = parser.parse_args()
    base_folder = args.folder[0]
    pano_width = args.width
    pano_height = args.height
    output_fn = args.output

    meta = MetaData.MetaData(base_folder)

    stitcher = Stitcher.Stitcher(base_folder, meta)
    timer = utils.Timer()
    print('- Load and preprocess images.....', end='', flush=True)
    stitcher.load_and_preprocess()
    print('{:.3f} seconds'.format(timer.end()))

    print('- Positioning images.....', end='')
    timer.begin()
    stitcher.position_frames()
    print('{:.3f} seconds'.format(timer.end()))

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


if __name__ == '__main__':
    main()

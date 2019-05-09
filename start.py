import argparse
import os
import shutil
import cv2 as cv
from libpano import MetaData
from libpano import Preprocess
from libpano import Stitcher
from libpano import Config
from libpano import utils

parser = argparse.ArgumentParser(prog='start.py', description='my panorama stitch program')
parser.add_argument('folder', nargs='+', help='folder containing files to stitch', type=str)
parser.add_argument('--width', help='Width of the output panorama', type=int, dest='width')
parser.add_argument('--height', help='Height of the output panorama', type=int, dest='height')
parser.add_argument('--output_file', default='output.jpg', help='File name of the output file.', type=str,
                    dest='output_file')

__doc__ = '\n' + parser.format_help()


def main():
    timer = utils.Timer()
    args = parser.parse_args()
    base_folder = args.folder[0]
    pano_width = args.width
    pano_height = args.height
    output_fn = args.output_file

    meta = MetaData.MetaData(base_folder)
    if Config.verbose:
        print("Meta data was loaded.")
        print(meta.metrics)

    # ###################################################### Pre-Processing image frames

    # make a temporary folder for work
    temp_folder = os.path.join(os.getcwd(), 'temp1')

    try:
        os.mkdir(temp_folder)
    except OSError:
        print('Creation of temp directory failed')
        exit(1)
    else:
        if Config.verbose:
            print('temp directory was created.')

    scale_x = pano_width / meta.metrics.PW
    scale_y = pano_height / meta.metrics.PH
    scale = max(scale_x, scale_y)

    # resize images
    # comment for dev
    print('Resizing images...')
    Preprocess.preprocess_resize(base_folder, temp_folder, meta, scale)

    # recalculate the metrics as the image size has changed
    meta.load_panorama_metrics(temp_folder)

    # transform images
    print('Transforming images...')
    Preprocess.preprocess_frames(temp_folder, meta)

    print('Positioning images...')
    stitcher = Stitcher.Stitcher(temp_folder, meta)

    print('Blending images...')
    stitcher.load_frames()
    output = stitcher.blend_frames()

    cv.imwrite(output_fn, output)

    print('Done in {:.2f} seconds.'.format(timer.end()))

    # delete of temp directory
    try:
        shutil.rmtree('temp')
    except OSError:
        print('Deletion of temp directory failed')
    else:
        print('temp directory was deleted.')


if __name__ == '__main__':
    print(__doc__)
    main()

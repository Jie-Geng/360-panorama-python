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
    if Config.debug:
        print("Meta data was loaded.")
        print(meta.metrics)

    # ###################################################### Pre-Processing image frames

    # make a temporary folder for work
    temp_folder = os.path.join(os.getcwd(), 'temp1')

    try:
        os.mkdir(temp_folder)
    except OSError:
        print('Creation of temp directory failed')
    else:
        if Config.debug:
            print('temp directory was created.')

    scale_x = pano_width / meta.metrics.PW
    scale_y = pano_height / meta.metrics.PH
    scale = max(scale_x, scale_y)

    # resize images
    # comment for dev
    if Config.debug:
        timer.begin()
        print('Resizing images scale={:.2f}...'.format(scale), end='', flush=True)

    Preprocess.preprocess_resize(base_folder, temp_folder, meta, scale)

    if Config.debug:
        print('Finished in {:.2f} seconds'.format(timer.end()))

    # recalculate the metrics as the image size has changed
    meta.load_panorama_metrics(temp_folder)
    print('meta was reloaded.')
    print(meta.metrics)

    manual_stitching_test = False
    if manual_stitching_test:
        row = 2
        fc = FocalCalculator.FocalCalculator(temp_folder, meta)
        focal = fc.get_focal(row, do_cylindrical_warp=False)
        df = meta.grid_data
        df = df[df.row == row].sort_values(by='yaw')
        uris = df['uri'].values.tolist()
        names = []
        focals = fc.focals
        for uri in uris:
            names.append(os.path.join(temp_folder, uri))

        manual.manual_main(names, focals)
        return

    # fc = FocalCalculator.FocalCalculator(temp_folder, meta)
    # for row in range(Config.start_row, Config.end_row):
    #     fc.get_focal(row, do_cylindrical_warp=True)

    # recalculate the metrics as the image size has changed
    meta.load_panorama_metrics(temp_folder)

    # """
    # transform images
    if Config.debug:
        timer.begin()
        print('Transforming images...', end='', flush=True)

    Preprocess.preprocess_warp(temp_folder, meta)

    if Config.debug:
        print('Finished in {:.2f} seconds'.format(timer.end()))

    return

    # begin stitching
    if Config.debug:
        timer.begin()
        print('Positioning images...', end='', flush=True)

    stitcher = Stitcher.Stitcher(temp_folder, meta)
    stitcher.load_frames()

    if Config.debug:
        print('Finished in {:.2f} seconds'.format(timer.end()))

    # blending images
    if Config.debug:
        timer.begin()
        print('Blending images...', end='', flush=True)

    output = stitcher.blend_frames()
    cv.imwrite(output_fn, output)

    if Config.debug:
        print('Finished in {:.2f} seconds'.format(timer.end()))

    # """

    # delete of temp directory
    try:
        shutil.rmtree('temp')
    except OSError:
        print('Deletion of temp directory failed')
    else:
        print('temp directory was deleted.')

    print('\nDone in {:.2f} seconds'.format(all_timer.end()))


if __name__ == '__main__':
    main()

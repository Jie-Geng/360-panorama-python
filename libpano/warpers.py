
import math
import numpy as np
import cv2 as cv
from scipy.interpolate import griddata

from libpano.ImageCropper import ImageCropper
from libpano import Config


def interpolate_image(image, mask):
    """
    Interpolate an image and its mask after warping
    This should be called only when it is needed, as it takes too long
    :param image: image array
    :param mask: mask array
    :return: (image, mask) interpolated
    """
    # print(image.shape, mask.shape)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    indices = np.where(gray != 0)

    nx, ny = image.shape[1], image.shape[0]
    x, y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))

    samples = image[indices]
    int_im = griddata(indices, samples, (y, x))

    blurred_mask = cv.GaussianBlur(mask, (25, 25), 0, 0)
    _, blurred_mask = cv.threshold(cv.cvtColor(blurred_mask, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)

    final_mask = cv.cvtColor(blurred_mask, cv.COLOR_GRAY2BGR)
    final_mask = cv.erode(final_mask, None)

    # print(int_im.shape, final_mask.shape)
    #
    int_im = cv.resize(int_im, (final_mask.shape[1], final_mask.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
    final_image = int_im * (final_mask / 255)

    del gray, samples, int_im, blurred_mask

    return final_image, final_mask


def spherical_warp(image, pitch, metrics):

    """
    Spherical warping not using focal length.
    It uses only FOV(field of view).

    It converts pixel coordinates into angle coordinate of latitude(-pi/2 ~ pi/2) and longitude(-pi ~ pi).

                              ^
                              | -pi/2
        ----------------------+----------------------
        |                     |                     |
        |                     |                     |
        |                     |                     |
        |                     |                     |
        |                     |                     |
    --------------------------+--------------------------->
    -pi |                     | 0                   | pi
        |                     |                     |
        |                     |                     |
        |                     |                     |
        |                     |                     |
        ----------------------+----------------------
                              | pi/2

    :param image: image ndarray
    :param pitch: pitch of the image center in radian
    :param metrics: camera metrics. It is needed to get fov and panorama size
    :return: (warped_image, mask_image)
    """

    # define constants and metrics
    pi2 = math.pi * 2
    pi_2 = math.pi / 2

    image_height, image_width, image_channel = image.shape

    pano_height = int(metrics.PH)
    pano_width = int(metrics.PW)
    pano_channel = image_channel

    # limit panorama size, as there appears black gaps when it is too big
    # pano_height = min(pano_height, 2048)
    # pano_width = min(pano_width, 4096)

    # for development
    # pano_height = Config.internal_panorama_height
    # pano_width = Config.internal_panorama_width

    # camera center point(in radian)
    center_point = np.array([0, pitch])

    fov = np.array([metrics.AOV_h / pi2, metrics.AOV_v / math.pi], np.float)

    # generate image map
    xx, yy = np.meshgrid(np.linspace(0, 1, image_width), np.linspace(0, 1, image_height))
    image_map = np.array([xx.ravel(), yy.ravel()]).T

    # convert into radian coordinate
    image_map = (image_map * 2 - 1) * np.array([math.pi, pi_2]) * (np.ones_like(image_map) * fov)

    # Calculate spherical coordinates
    #
    # This algorithm is described in this great blog
    # https://http://blog.nitishmutha.com/equirectangular/360degree/2017/06/12/How-to-project-Equirectangular-image-to-rectilinear-view.html
    #

    x = image_map.T[0]
    y = image_map.T[1]

    rou = np.sqrt(x ** 2 + y ** 2)
    c = np.arctan(rou)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    lat = np.arcsin(cos_c * np.sin(center_point[1]) + (y * sin_c * np.cos(center_point[1])) / rou)
    lon = center_point[0] + \
        np.arctan2(x * sin_c, rou * np.cos(center_point[1]) * cos_c - y * np.sin(center_point[1]) * sin_c)

    lat = (lat / pi_2 + 1.) * 0.5
    lon = (lon / math.pi + 1.) * 0.5

    # Mapping image frame into spherical space
    #

    # convert radian coordinates into pixel coordinates
    map_x = np.mod(lon, 1) * pano_width
    map_y = np.mod(lat, 1) * pano_height

    map_x = np.floor(map_x).astype(int)
    map_y = np.floor(map_y).astype(int)

    # flatten image and copy data
    flat_idx = map_y * pano_width + map_x
    del image_map, lat, lon, map_x, map_y

    warped = np.zeros((pano_height, pano_width, pano_channel), np.float)
    warped = np.reshape(warped, [-1, pano_channel])

    flat_img = np.reshape(image, [-1, image_channel])
    warped[flat_idx] = flat_img

    # mask image process
    mask_img = np.ones_like(flat_img, np.float) * 255
    mask_warped = np.zeros_like(warped, np.float)
    mask_warped[flat_idx] = mask_img

    # reshape into their original shape
    warped = np.reshape(warped, [pano_height, pano_width, pano_channel]).astype(np.uint8)
    mask_warped = np.reshape(mask_warped, [pano_height, pano_width, pano_channel]).astype(np.uint8)
    # warped[others_dest] = flat_img[others_src]

    # crop images
    image_cropper = ImageCropper(warped, max_border_size=0)
    warped = image_cropper.crop()

    mask_cropper = ImageCropper(mask_warped, max_border_size=0)
    mask_warped = mask_cropper.crop()

    del image, flat_img, mask_img, image_cropper, mask_cropper

    # if abs(pitch) >= 0.75:
    #     warped, mask_warped = interpolate_image(warped, mask_warped)
    #
    # return warped.astype(np.uint8), mask_warped.astype(np.uint8)
    return interpolate_image(warped, mask_warped)


def cylindrical_warp_with_focal(img, focal_length):
    """
    This functions performs cylindrical warping, but its speed is slow and deprecated.

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


def cylindrical_warp_with_k(img, k):
    """
    This function returns the cylindrical warp for a given image and intrinsics matrix K
    """

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

    img = cv.remap(img,
                   b[:, :, 0].astype(np.float32),
                   b[:, :, 1].astype(np.float32),
                   cv.INTER_AREA,
                   borderMode=cv.BORDER_CONSTANT,
                   borderValue=(0, 0, 0))

    # Crop black border
    cropper = ImageCropper(img)
    cropped_image = cropper.crop()

    return cropped_image


def render_image_to_canvas(canvas, image, pitch, yaw, metrics):

    """
    Spherical warping not using focal length.
    It uses only FOV(field of view).

    It converts pixel coordinates into angle coordinate of latitude(-pi/2 ~ pi/2) and longitude(-pi ~ pi).

                              ^
                              | -pi/2
        ----------------------+----------------------
        |                     |                     |
        |                     |                     |
        |                     |                     |
        |                     |                     |
        |                     |                     |
    --------------------------+--------------------------->
    -pi |                     | 0                   | pi
        |                     |                     |
        |                     |                     |
        |                     |                     |
        |                     |                     |
        ----------------------+----------------------
                              | pi/2

    :param canvas: image canvas for panorama
    :param image: image ndarray
    :param pitch: pitch of the image center in radian
    :param yaw: yaw of the image center in radian
    :param metrics: camera metrics. It is needed to get fov and panorama size
    :return: (warped_image, mask_image)
    """

    # define constants and metrics
    pi2 = math.pi * 2
    pi_2 = math.pi / 2

    image_height, image_width, image_channel = image.shape

    pano_height, pano_width, pano_channel = canvas.shape

    # camera center point(in radian)
    center_point = np.array([yaw, pitch])

    fov = np.array([metrics.AOV_h / pi2, metrics.AOV_v / math.pi], np.float)

    # generate image map
    xx, yy = np.meshgrid(np.linspace(0, 1, image_width), np.linspace(0, 1, image_height))
    image_map = np.array([xx.ravel(), yy.ravel()]).T

    # convert into radian coordinate
    image_map = (image_map * 2 - 1) * np.array([math.pi, pi_2]) * (np.ones_like(image_map) * fov)

    # Calculate spherical coordinates
    #
    # This algorithm is described in this great blog
    # https://http://blog.nitishmutha.com/equirectangular/360degree/2017/06/12/How-to-project-Equirectangular-image-to-rectilinear-view.html
    #

    x = image_map.T[0]
    y = image_map.T[1]

    rou = np.sqrt(x ** 2 + y ** 2)
    c = np.arctan(rou)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    lat = np.arcsin(cos_c * np.sin(center_point[1]) + (y * sin_c * np.cos(center_point[1])) / rou)
    lon = center_point[0] + \
        np.arctan2(x * sin_c, rou * np.cos(center_point[1]) * cos_c - y * np.sin(center_point[1]) * sin_c)

    lat = (lat / pi_2 + 1.) * 0.5
    lon = (lon / math.pi + 1.) * 0.5

    # Mapping image frame into spherical space
    #

    # convert radian coordinates into pixel coordinates
    map_x = np.mod(lon, 1) * pano_width
    map_y = np.mod(lat, 1) * pano_height

    map_x = np.floor(map_x).astype(int)
    map_y = np.floor(map_y).astype(int)

    # flatten image and copy data
    flat_idx = map_y * pano_width + map_x
    del image_map, lat, lon, map_x, map_y

    # warped = np.zeros((pano_height, pano_width, pano_channel), np.float)
    canvas = np.reshape(canvas, [-1, pano_channel])

    flat_img = np.reshape(image, [-1, image_channel])
    canvas[flat_idx] = flat_img

    temp_canvas = np.zeros_like(canvas, np.uint8)
    temp_canvas[flat_idx] = flat_img
    temp_canvas = np.reshape(temp_canvas, [pano_height, pano_width, pano_channel]).astype(np.uint8)

    cropper = ImageCropper(temp_canvas, max_border_size=1)
    temp_canvas = cropper.crop()

    # reshape into their original shape
    canvas = np.reshape(canvas, [pano_height, pano_width, pano_channel]).astype(np.uint8)

    del image, flat_img

    return canvas, temp_canvas

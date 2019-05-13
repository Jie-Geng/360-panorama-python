
import math
import numpy as np
import cv2 as cv
from libpano.ImageCropper import ImageCropper


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
    pano_height = min(pano_height, 2048)
    pano_width = min(pano_width, 4096)

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
    # TODO: interpolation for near-pole areas(arctic and antarctic)

    # convert radian coordinates into pixel coordinates
    map_x = np.mod(lon, 1) * pano_width
    map_y = np.mod(lat, 1) * pano_height

    map_x = np.floor(map_x).astype(int)
    map_y = np.floor(map_y).astype(int)

    # flatten image and copy data
    flat_idx = map_y * pano_width + map_x

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

    # crop images
    image_cropper = ImageCropper(warped, max_border_size=0)
    mask_cropper = ImageCropper(mask_warped, max_border_size=0)

    return image_cropper.crop(), mask_cropper.crop()


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

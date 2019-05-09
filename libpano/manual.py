import cv2
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from libpano import FocalCalculator

DEBUG             = True
FEATURE_THRESHOLD = 0.01
DESCRIPTOR_SIZE   = 5
MATCHING_Y_RANGE  = 50

RANSAC_K          = 200
RANSAC_THRES_DISTANCE = 3

ALPHA_BLEND_WINDOW = 20

FEATURE_CUT_X_EDGE = 5
FEATURE_CUT_Y_EDGE = 5


def compute_r(xx_row, yy_row, xy_row, k):
    row_response = np.zeros(shape=xx_row.shape, dtype=np.float32)
    for x in range(len(xx_row)):
        det_M = xx_row[x] * yy_row[x] - xy_row[x] ** 2
        trace_M = xx_row[x] + yy_row[x]
        R = det_M - k * trace_M ** 2
        row_response[x] = R

    return row_response


"""
Harris corner detector

Args:
    img: input image
    pool: for multiprocessing
    k: harris corner constant value
    block_size: harris corner windows size

Returns:
    A corner response matrix. width, height same as input image
"""


def harris_corner(img, pool, k=0.08, block_size=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255

    corner_response = np.zeros(shape=gray.shape, dtype=np.float32)

    height, width, _ = img.shape
    dx = cv2.Sobel(gray, -1, 1, 0)
    dy = cv2.Sobel(gray, -1, 0, 1)
    Ixx = dx * dx
    Iyy = dy * dy
    Ixy = dx * dy

    cov_xx = cv2.boxFilter(Ixx, -1, (block_size, block_size), normalize=False)
    cov_yy = cv2.boxFilter(Iyy, -1, (block_size, block_size), normalize=False)
    cov_xy = cv2.boxFilter(Ixy, -1, (block_size, block_size), normalize=False)

    corner_response = pool.starmap(compute_r, [(cov_xx[y], cov_yy[y], cov_xy[y], k) for y in range(height)])

    return np.asarray(corner_response)


"""
Extract descritpor from corner response image

Args:
    corner_response: corner response matrix
    threshlod: only corner response > 'max_corner_response*threshold' will be extracted
    kernel: descriptor's window size, the descriptor will be kernel^2 dimension vector 

Returns:
    A pair of (descriptors, positions)
"""


def extract_description(img, corner_response, threshold=0.05, kernel=3):
    height, width = corner_response.shape

    # Reduce corner
    features = np.zeros(shape=(height, width), dtype=np.uint8)
    features[corner_response > threshold * corner_response.max()] = 255

    # Trim feature on image edge
    features[:FEATURE_CUT_Y_EDGE, :] = 0
    features[-FEATURE_CUT_Y_EDGE:, :] = 0
    features[:, -FEATURE_CUT_X_EDGE:] = 0
    features[:, :FEATURE_CUT_X_EDGE] = 0

    # Reduce features using local maximum
    window = 3
    for y in range(0, height - 10, window):
        for x in range(0, width - 10, window):
            if features[y:y + window, x:x + window].sum() == 0:
                continue
            block = corner_response[y:y + window, x:x + window]
            max_y, max_x = np.unravel_index(np.argmax(block), (window, window))
            features[y:y + window, x:x + window] = 0
            features[y + max_y][x + max_x] = 255

    feature_positions = []
    feature_descriptions = np.zeros(shape=(1, kernel ** 2), dtype=np.float32)

    half_k = kernel // 2
    for y in range(half_k, height - half_k):
        for x in range(half_k, width - half_k):
            if features[y][x] == 255:
                feature_positions += [[y, x]]
                desc = corner_response[y - half_k:y + half_k + 1, x - half_k:x + half_k + 1]
                feature_descriptions = np.append(feature_descriptions, [desc.flatten()], axis=0)

    return feature_descriptions[1:], feature_positions


"""
Matching two groups of descriptors

Args:
    descriptor1:
    descriptor2:
    feature_position1: descriptor1's corrsponsed position
    feature_position2: descriptor2's corrsponsed position
    pool: for mulitiprocessing
    y_range: restrict only to match y2-y_range < y < y2+y_range

Returns:
    matched position pairs, it is a Nx2x2 matrix
"""


def matching(descriptor1, descriptor2, feature_position1, feature_position2, pool, y_range=10):
    TASKS_NUM = 32

    partition_descriptors = np.array_split(descriptor1, TASKS_NUM)
    partition_positions = np.array_split(feature_position1, TASKS_NUM)

    sub_tasks = [(partition_descriptors[i], descriptor2, partition_positions[i], feature_position2, y_range) for i in
                 range(TASKS_NUM)]
    results = pool.starmap(compute_match, sub_tasks)

    matched_pairs = []
    for res in results:
        if len(res) > 0:
            matched_pairs += res

    return matched_pairs


def compute_match(descriptor1, descriptor2, feature_position1, feature_position2, y_range=10):
    matched_pairs = []
    matched_pairs_rank = []

    for i in range(len(descriptor1)):
        distances = []
        y = feature_position1[i][0]
        for j in range(len(descriptor2)):
            diff = float('Inf')

            # only compare features that have similar y-axis
            if y - y_range <= feature_position2[j][0] <= y + y_range:
                diff = descriptor1[i] - descriptor2[j]
                diff = (diff ** 2).sum()
            distances += [diff]

        sorted_index = np.argpartition(distances, 1)
        local_optimal = distances[sorted_index[0]]
        local_optimal2 = distances[sorted_index[1]]
        if local_optimal > local_optimal2:
            local_optimal, local_optimal2 = local_optimal2, local_optimal

        if local_optimal / local_optimal2 <= 0.5:
            paired_index = np.where(distances == local_optimal)[0][0]
            pair = [feature_position1[i], feature_position2[paired_index]]
            matched_pairs += [pair]
            matched_pairs_rank += [local_optimal]

    # Refine pairs
    sorted_rank_idx = np.argsort(matched_pairs_rank)
    sorted_match_pairs = np.asarray(matched_pairs)
    sorted_match_pairs = sorted_match_pairs[sorted_rank_idx]

    refined_matched_pairs = []
    for item in sorted_match_pairs:
        duplicated = False
        for refined_item in refined_matched_pairs:
            if refined_item[1] == list(item[1]):
                duplicated = True
                break
        if not duplicated:
            refined_matched_pairs += [item.tolist()]

    return refined_matched_pairs


"""
Find best shift using RANSAC

Args:
    matched_pairs: matched pairs of feature's positions, its an Nx2x2 matrix
    prev_shift: previous shift, for checking shift direction.

Returns:
    Best shift [y x]. ex. [4 234]

Raise:
    ValueError: Shift direction NOT same as previous shift.
"""


def RANSAC(matched_pairs, prev_shift):
    matched_pairs = np.asarray(matched_pairs)

    use_random = True if len(matched_pairs) > RANSAC_K else False

    best_shift = []
    K = RANSAC_K if use_random else len(matched_pairs)
    threshold_distance = RANSAC_THRES_DISTANCE

    max_inliner = 0
    for k in range(K):
        # Random pick a pair of matched feature
        idx = int(np.random.random_sample() * len(matched_pairs)) if use_random else k
        sample = matched_pairs[idx]

        # fit the warp model
        shift = sample[1] - sample[0]

        # calculate inliner points
        shifted = matched_pairs[:, 1] - shift
        difference = matched_pairs[:, 0] - shifted

        inliner = 0
        for diff in difference:
            if np.sqrt((diff ** 2).sum()) < threshold_distance:
                inliner = inliner + 1

        if inliner > max_inliner:
            max_inliner = inliner
            best_shift = shift

    if prev_shift[1] * best_shift[1] < 0:
        print('\n\nBest shift:', best_shift)
        raise ValueError('Shift direction NOT same as previous shift.')

    return best_shift


"""
Stitch two image with blending.

Args:
    img1: first image
    img2: second image
    shift: the relative position between img1 and img2
    pool: for multiprocessing
    blending: using blending or not

Returns:
    A stitched image
"""


def stitching(img1, img2, shift, pool, blending=True):
    padding = [
        (shift[0], 0) if shift[0] > 0 else (0, -shift[0]),
        (shift[1], 0) if shift[1] > 0 else (0, -shift[1]),
        (0, 0)
    ]
    shifted_img1 = np.lib.pad(img1, padding, 'constant', constant_values=0)

    # cut out unnecessary region
    split = img2.shape[1] + abs(shift[1])
    splited = shifted_img1[:, split:] if shift[1] > 0 else shifted_img1[:, :-split]
    shifted_img1 = shifted_img1[:, :split] if shift[1] > 0 else shifted_img1[:, -split:]

    h1, w1, _ = shifted_img1.shape
    h2, w2, _ = img2.shape

    inv_shift = [h1 - h2, w1 - w2]
    inv_padding = [
        (inv_shift[0], 0) if shift[0] < 0 else (0, inv_shift[0]),
        (inv_shift[1], 0) if shift[1] < 0 else (0, inv_shift[1]),
        (0, 0)
    ]
    shifted_img2 = np.lib.pad(img2, inv_padding, 'constant', constant_values=0)

    direction = 'left' if shift[1] > 0 else 'right'

    if blending:
        seam_x = shifted_img1.shape[1] // 2
        tasks = [(shifted_img1[y], shifted_img2[y], seam_x, ALPHA_BLEND_WINDOW, direction) for y in range(h1)]
        shifted_img1 = pool.starmap(alpha_blend, tasks)
        shifted_img1 = np.asarray(shifted_img1)
        shifted_img1 = np.concatenate((shifted_img1, splited) if shift[1] > 0 else (splited, shifted_img1), axis=1)
    else:
        raise ValueError('I did not implement "blending=False" ^_^')

    return shifted_img1


def alpha_blend(row1, row2, seam_x, window, direction='left'):
    if direction == 'right':
        row1, row2 = row2, row1

    new_row = np.zeros(shape=row1.shape, dtype=np.uint8)

    for x in range(len(row1)):
        color1 = row1[x]
        color2 = row2[x]
        if x < seam_x - window:
            new_row[x] = color2
        elif x > seam_x + window:
            new_row[x] = color1
        else:
            ratio = (x - seam_x + window) / (window * 2)
            new_row[x] = (1 - ratio) * color2 + ratio * color1

    return new_row


"""
End to end alignment

Args:
    img: panoramas image
    shifts: all shifts for each image in panoramas

Returns:
    A image that fixed the y-asix shift error
"""


def end2end_align(img, shifts):
    sum_y, sum_x = np.sum(shifts, axis=0)

    y_shift = np.abs(sum_y)
    col_shift = None

    # same sign
    if sum_x * sum_y > 0:
        col_shift = np.linspace(y_shift, 0, num=img.shape[1], dtype=np.uint16)
    else:
        col_shift = np.linspace(0, y_shift, num=img.shape[1], dtype=np.uint16)

    aligned = img.copy()
    for x in range(img.shape[1]):
        aligned[:, x] = np.roll(img[:, x], col_shift[x], axis=0)

    return aligned


"""
Crop the black border in image

Args:
    img: a panoramas image

Returns:
    Cropped image
"""


def crop(img):
    _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    upper, lower = [-1, -1]

    black_pixel_num_threshold = img.shape[1] // 100

    for y in range(thresh.shape[0]):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
            upper = y
            break

    for y in range(thresh.shape[0] - 1, 0, -1):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
            lower = y
            break

    return img[upper:lower, :]


def matched_pairs_plot(p1, p2, mp):
    _, offset, _ = p1.shape
    plt_img = np.concatenate((p1, p2), axis=1)
    plt.figure(figsize=(10,10))
    plt.imshow(plt_img)
    for i in range(len(mp)):
        plt.scatter(x=mp[i][0][1], y=mp[i][0][0], c='r')
        plt.plot([mp[i][0][1], offset+mp[i][1][1]], [mp[i][0][0], mp[i][1][0]], 'y-', lw=1)
        plt.scatter(x=offset+mp[i][1][1], y=mp[i][1][0], c='b')
    plt.show()
    cv2.waitKey(0)


def manual_main(file_names, focal_lengths):

    pool = mp.Pool(mp.cpu_count())

    print('Warp images to cylinder')
    img_list = []
    ks = []
    args = []
    for idx in range(len(focal_lengths)):
        img = cv2.imread(file_names[idx])
        img_list.append(img)

        focal = focal_lengths[idx]
        h, w = img.shape[:2]
        k = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])  # mock intrinsics
        ks.append(k)

        args.append((img, k))

    cylinder_img_list = pool.starmap(FocalCalculator.FocalCalculator.cylindrical_warp, args)

    _, img_width, _ = img_list[0].shape
    stitched_image = cylinder_img_list[0].copy()

    shifts = [[0, 0]]
    cache_feature = [[], []]

    # add first img for end to end align
    # cylinder_img_list += [stitched_image]
    for i in range(1, len(cylinder_img_list)):
        print('Computing .... ' + str(i + 1) + '/' + str(len(cylinder_img_list)))
        img1 = cylinder_img_list[i - 1]
        img2 = cylinder_img_list[i]

        print(' - Find features in previous img .... ', end='', flush=True)
        descriptors1, position1 = cache_feature
        if len(descriptors1) == 0:
            corner_response1 = harris_corner(img1, pool)
            descriptors1, position1 = extract_description(img1, corner_response1, kernel=DESCRIPTOR_SIZE,
                                                          threshold=FEATURE_THRESHOLD)
        print(str(len(descriptors1)) + ' features extracted.')

        print(' - Find features in img_' + str(i + 1) + ' .... ', end='', flush=True)
        corner_response2 = harris_corner(img2, pool)
        descriptors2, position2 = extract_description(img2, corner_response2, kernel=DESCRIPTOR_SIZE,
                                                      threshold=FEATURE_THRESHOLD)
        print(str(len(descriptors2)) + ' features extracted.')

        cache_feature = [descriptors2, position2]

        if DEBUG:
            cv2.imshow('cr1', corner_response1)
            cv2.imshow('cr2', corner_response2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(' - Feature matching .... ', end='', flush=True)
        matched_pairs = matching(descriptors1, descriptors2, position1, position2, pool,
                                 y_range=MATCHING_Y_RANGE)
        print(str(len(matched_pairs)) + ' features matched.')

        if DEBUG:
            matched_pairs_plot(img1, img2, matched_pairs)

        # filtering matched pairs
        filtered = []
        height1 = img1.shape[0]
        width1 = img1.shape[1]
        width2 = img2.shape[1]
        for pair in matched_pairs:
            if pair[0][1] < width1 / 2:
                continue
            if pair[1][1] > width2 / 2:
                continue
            width = pair[1][1] + width1 - pair[0][1]
            if width > 4 * width1 / 5:
                continue

            if abs(pair[0][0] - pair[1][0]) > height1 / 5:
                continue

            filtered.append(pair)

        print(str(len(filtered)) + ' matched were filtered.')
        matched_pairs = filtered

        if DEBUG:
            matched_pairs_plot(img1, img2, matched_pairs)

        print(' - Find best shift using RANSAC .... ', end='', flush=True)
        shift = RANSAC(matched_pairs, shifts[-1])
        shifts += [shift]
        print('best shift ', shift)

        print(' - Stitching image .... ', end='', flush=True)
        stitched_image = stitching(stitched_image, img2, shift, pool, blending=True)
        cv2.imwrite(str(i) + '.jpg', stitched_image)
        print('Saved.')

    print('Perform end to end alignment')
    aligned = end2end_align(stitched_image, shifts)
    cv2.imwrite('aligned.jpg', aligned)

    print('Cropping image')
    cropped = crop(aligned)
    cv2.imwrite('cropped.jpg', cropped)

import os
import numpy as np
import matplotlib.pyplot as plt
import time

from collections import defaultdict
from skimage.io import imread
from skimage.feature import canny
from skimage.filters import sobel

THETA_NUMBERS = 60 # MUST BE LESS OR EQUAL THAN 360
threshold = 0.1

def edges_gradient(image):
    """
    Calculates edges and gradient orientations for the image

    :param image: image, represented by 2D array
    :type image: numpy.ndarray
    """

    edges = canny(image, mode='nearest', sigma=1)
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    gradient = np.mod(np.round(np.arctan2(-dx, dy) * THETA_NUMBERS / (2 * np.pi)), THETA_NUMBERS)

    # fig, ax = plt.subplots(2, 2)
    # ax = ax.ravel()
    # ax[0].imshow(-dx, cmap='gray')
    # ax[1].imshow(dy, cmap='gray')
    # ax[2].imshow(gradient, cmap='gray')
    # plt.savefig('images/dxdy.png')

    return edges, gradient

def build_r_table(reference_image):
    """
    Builds the R-table from the given shape image

    :param reference_image: reference shape to be found on images
    :type reference_image: numpy.ndarray
    """

    origin = (int(reference_image.shape[0] / 2), int(reference_image.shape[1] / 2)) # origin is the center of the image
    edges, gradient = edges_gradient(reference_image)

    # fig, ax = plt.subplots(2, 1)
    # ax[0].imshow(edges, cmap='gray')
    # ax[1].imshow(gradient, cmap='gray')
    # plt.savefig('images/ref.png')


    r_table = defaultdict(list)
    for (i, j), value in np.ndenumerate(edges):
        if value: # if this is an edge point
            # assign radius vector (from this point to the origin) to the gradient value at this point
            r_table[gradient[i, j]].append((origin[0] - i, origin[1] - j))

    return r_table

def build_accumulator(r_table, query_image):
    """
    Builds the accumulator array for a given image using the R-table

    :param r_table: R-table for the reference shape that we search for
    :type r_table: dict
    :param query_image: image that presumably contains reference shape
    :type query_image: numpy.ndarray
    """

    edges, gradient = edges_gradient(query_image)

    accumulator = np.zeros((THETA_NUMBERS, *query_image.shape))

    # fig, ax = plt.subplots(2, 1)
    # ax[0].imshow(edges, cmap='gray')
    # ax[1].imshow(gradient, cmap='gray')
    # plt.savefig('images/que.png')

    for (i, j), value in np.ndenumerate(edges):
        if value: # if this is an edge point
            for theta in np.arange(0, THETA_NUMBERS, dtype=int): # for every point that has a certain gradient value
                for r in r_table[np.mod(gradient[i, j] - theta, THETA_NUMBERS)]:
                    accum_i = i + r[0] * np.cos(theta * 2 * np.pi / THETA_NUMBERS) - r[1] * np.sin(theta * 2 * np.pi / THETA_NUMBERS)
                    accum_j = j + r[0] * np.sin(theta * 2 * np.pi / THETA_NUMBERS) + r[1] * np.cos(theta * 2 * np.pi / THETA_NUMBERS)
                    # and increase the relative value by one
                    if accum_i < accumulator.shape[1] and accum_j < accumulator.shape[2]:
                        accumulator[int(theta), int(accum_i), int(accum_j)] += 1


    accumulator_view = np.sum(accumulator, axis=0)

    return accumulator, accumulator_view

def n_max_ind(array, n):
    """
    Returns the indices of n maximum values in an N-dimensional array

    :param array: N-dimensional array
    :type array: numpy.ndarray
    :param n: the desired number of maximum values
    :type n: int
    """
    if n == 0:
        return np.asarray([])
    indices = array.ravel().argsort()[-n:]
    indices = np.unravel_index(indices, array.shape)
    return np.stack(indices, axis=1)

def general_hough(reference_image, query_image, path_for_saving):
    r_table = build_r_table(reference_image)
    accumulator, accumulator_view = build_accumulator(r_table, query_image)

    max_accum_ = np.max(accumulator, axis=(1, 2))
    print(max_accum_)

    plt.gray()
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = ax.ravel()
    ax1.set_title('Reference image')
    ax1.imshow(reference_image)
    ax2.set_title('Suspicious image')
    ax2.imshow(query_image)
    ax3.set_title('Accumulator')
    ax3.imshow(accumulator_view)
    ax4.set_title('Detection')
    ax4.imshow(query_image)

    max_accum = np.max(accumulator)
    for theta, accum in enumerate(accumulator):
        greater = np.where(accum > threshold * max_accum, accum, 0)
        top = n_max_ind(greater, min(np.count_nonzero(greater), 4))
        print(top)
        for v_coord, h_coord in top:
            dx, dy = np.cos(theta * 2 * np.pi / THETA_NUMBERS), np.sin(theta * 2 * np.pi / THETA_NUMBERS)
            ax4.arrow(h_coord, v_coord, 100 * dy, 100 * dx, width=2, head_width=0, color='y')
            # ax4.scatter(h_coord, v_coord, marker='o', color='r', s=10)


    plt.savefig(path_for_saving)


def test(reference_path="./images/line.png", query_path="./images/line_test5.png"):
    reference_image = imread(reference_path, as_gray=True)
    query_image = imread(query_path, as_gray=True)
    d, f = os.path.split(query_path)[0], os.path.splitext(os.path.split(query_path)[1])[0]
    path_for_saving = os.path.join(d, f + '_R.png')

    general_hough(reference_image, query_image, path_for_saving)


if __name__ == '__main__':
    plt.clf()
    test()

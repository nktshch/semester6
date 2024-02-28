import os
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from skimage.io import imread
from skimage.feature import canny
from skimage.filters import sobel

theta_numbers = 360 # DO NOT CHANGE
threshold = 3

def gradient_orientation(edges_image):
    """
    Calculates the gradient orientations for edge points in the boolean image containing edges

    :param edges_image: 2-dimensional boolean array representing edges
    :type edges_image: numpy.ndarray
    """

    dx = sobel(edges_image, axis=0, mode='reflect')
    dy = sobel(edges_image, axis=1, mode='reflect')
    gradient = np.arctan2(dy, dx) + np.pi

    return np.round(np.rad2deg(gradient))

def build_r_table(shape_image):
    """
    Builds the R-table from the given shape image

    :param shape_image: reference shape to be found on images
    :type shape_image: numpy.ndarray
    """

    origin = (int(shape_image.shape[0] / 2), int(shape_image.shape[1] / 2)) # origin is the center of the image
    edges = canny(shape_image, mode='reflect')
    gradient = gradient_orientation(edges)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(gradient, cmap='gray')
    plt.savefig('images/9_gradient.png')
    r_table = defaultdict(list)
    for (i, j), value in np.ndenumerate(edges):
        if value: # if this is an edge point
            # assign radius vector (from this point to the origin) to the gradient value at this point
            r_table[gradient[i, j]].append((origin[0] - i, origin[1] - j))

    return r_table

def build_accumulator(r_table, image):
    """
    Builds the accumulator array for a given image using the R-table

    :param r_table: R-table for the reference shape that we search for
    :type r_table: dict
    :param image: image that presumably contains reference shape
    :type image: numpy.ndarray
    """

    edges = canny(image, mode='nearest')
    gradient = gradient_orientation(edges)
    accumulator = np.zeros((*image.shape, theta_numbers))
    accumulator_view = np.zeros(image.shape)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(edges, cmap='gray')
    plt.savefig('images/9_im_edges.png')

    for (i, j), value in np.ndenumerate(edges):
        if value: # if this is an edge point
            for theta in np.arange(0., 360., 1.0):
                # for every point that has a certain gradient value

                for r in r_table[np.mod(gradient[i, j] - theta, 360)]:
                    # calculate the position of supposed origin
                    # print(f"theta = {theta}")
                    accum_i = i - np.round(r[0] * np.cos(np.deg2rad(theta)) + r[1] * np.sin(np.deg2rad(theta))).astype(int)
                    accum_j = j - np.round(r[0] * np.sin(np.deg2rad(theta)) - r[1] * np.cos(np.deg2rad(theta))).astype(int)
                    # and increase the relative value by one
                    if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
                        # print(f"theta = {theta}")
                        accumulator[accum_i, accum_j, int(theta)] += 1
                        accumulator_view[accum_i, accum_j] += 1

    return accumulator, accumulator_view

def n_max(array, n):
    """
    Returns the indices of n maximum values in a 3-dimensional array

    :param array: 3-dimensional array
    :type array: numpy.ndarray
    :param n: the desired number of maximum values
    :type n: int
    """

    indices = array.ravel().argsort()[-n:]
    indices = np.unravel_index(indices, array.shape)

    return list(zip(indices[0], indices[1], indices[2]))

def test_general_hough(r_table, reference_image, query):
    query_image = imread(query, as_gray=True)
    accumulator, accumulator_view = build_accumulator(r_table, query_image)

    # for theta_as_dim in np.arange(0, 360, 1):
    #     print(f"{np.max(accumulator[:, :, theta_as_dim])} for {theta_as_dim}")

    plt.clf()
    plt.gray()
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # ax1, ax4 = ax.ravel()
    ax1, ax2, ax3, ax4 = ax.ravel()

    ax1.set_title('Reference image')
    ax1.imshow(reference_image)

    ax2.set_title('Suspicious image')
    ax2.imshow(query_image)

    ax3.set_title('Accumulator')
    ax3.imshow(accumulator_view)

    ax4.set_title('Detection')
    ax4.imshow(query_image)

    top = np.argwhere(accumulator > threshold)
    # print(top)
    vertical_coords = [point[0] for point in top]
    horizontal_coords = [point[1] for point in top]
    angles = [point[2] for point in top]

    for index, angle in enumerate(np.deg2rad(angles)):
        y_0, x_0 = vertical_coords[index], horizontal_coords[index]
        dx, dy = np.cos(angle - np.pi), np.sin(angle - np.pi)
        ax4.arrow(x_0, y_0, 10 * dx, 10 * dy, width=0.05, color='y')
    # ax4.scatter(horizontal_coords, vertical_coords, marker='o', color='r', s=10)


    d, f = os.path.split(query)[0], os.path.splitext(os.path.split(query)[1])[0]
    plt.savefig(os.path.join(d, f + '_noedges_output_.png'))


def test():
    # reference_image = imread("./images/line.png", as_gray=True)
    reference_image = np.vstack((np.ones((4, 15)), np.zeros((10, 15)), np.ones((4, 15)))).transpose(1, 0)

    # fig, ax = plt.subplots(figsize=(12, 4))
    # ax.set_title("--")
    # ax.imshow(reference_image, cmap='gray')
    # plt.show()

    r_table = build_r_table(reference_image)

    test_general_hough(r_table, reference_image, "./images/line10_test1.png")






if __name__ == '__main__':
    plt.clf()
    test()

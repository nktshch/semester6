"""General Hough Transform."""

from collections import defaultdict
import os

import click
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny
from skimage.filters import sobel
from skimage.io import imread


from . import __version__

THETA_NUMBERS = None


def edges_gradient(image: np.ndarray) -> tuple:
    """Calculates edges and gradient orientations for the image.

    Args:
        image: Image, represented by 2D array

    Returns:
        Edges and gradient of the image, in a tuple.
    """
    edges = canny(image, mode="nearest", sigma=1)
    dx = sobel(image, axis=0, mode="constant")
    dy = sobel(image, axis=1, mode="constant")
    gradient = np.mod(
        np.round(np.arctan2(-dx, dy) * THETA_NUMBERS / (2 * np.pi)), THETA_NUMBERS
    )

    # fig, ax = plt.subplots(2, 2)
    # ax = ax.ravel()
    # ax[0].imshow(-dx, cmap='gray')
    # ax[1].imshow(dy, cmap='gray')
    # ax[2].imshow(gradient, cmap='gray')
    # plt.savefig('images/dxdy.png')

    return edges, gradient


def build_r_table(reference_image: np.ndarray) -> defaultdict:
    """Builds the R-table from the given image.

    Given image, builds R-table, which is a defaultdict with angles as keys and vectors to origin
    of the image (center point by default) as values.

    Args:
        reference_image: Reference shape to be found on images

    Returns:
        R_table, represented by defaultdict.
    """
    # origin is the center of the image
    origin = (int(reference_image.shape[0] / 2), int(reference_image.shape[1] / 2))
    edges, gradient = edges_gradient(reference_image)

    # fig, ax = plt.subplots(2, 1)
    # ax[0].imshow(edges, cmap='gray')
    # ax[1].imshow(gradient, cmap='gray')
    # plt.savefig('images/ref.png')

    r_table = defaultdict(list)
    for (i, j), value in np.ndenumerate(edges):
        if value:  # if this is an edge point
            # assign radius vector (from this point to the origin) to the gradient value at this point
            r_table[gradient[i, j]].append((origin[0] - i, origin[1] - j))

    return r_table


def build_accumulator(r_table: defaultdict, query_image: np.ndarray) -> tuple:
    """Builds the accumulator array for a given image using the R-table.

    Given R-table for a reference image, creates accumulator for another image.
    Both images should be grayscale. Also, creates accumulator that is used for plotting. Its elements
    are sums along the dimension corresponding to angles in the main accumulator.

    Args:
        r_table: R-table for the reference shape that we search for
        query_image: Image that presumably contains reference shape

    Returns:
        Tuple of the main accumulator (used for finding the most probable locations)
        and accumulator used for plotting.
    """
    edges, gradient = edges_gradient(query_image)

    accumulator = np.zeros((THETA_NUMBERS, *query_image.shape))

    # fig, ax = plt.subplots(2, 1)
    # ax[0].imshow(edges, cmap='gray')
    # ax[1].imshow(gradient, cmap='gray')
    # plt.savefig('images/que.png')

    for (i, j), value in np.ndenumerate(edges):
        if value:  # if this is an edge point
            # for every point that has a certain gradient value
            for theta in np.arange(0, THETA_NUMBERS, dtype=int):
                for r in r_table[np.mod(gradient[i, j] - theta, THETA_NUMBERS)]:
                    accum_i = (i + r[0] * np.cos(theta * 2 * np.pi / THETA_NUMBERS)
                               - r[1] * np.sin(theta * 2 * np.pi / THETA_NUMBERS))
                    accum_j = (j + r[0] * np.sin(theta * 2 * np.pi / THETA_NUMBERS)
                               + r[1] * np.cos(theta * 2 * np.pi / THETA_NUMBERS))
                    # and increase the relative value by one
                    if (
                        accum_i < accumulator.shape[1]
                        and accum_j < accumulator.shape[2]
                    ):
                        accumulator[int(theta), int(accum_i), int(accum_j)] += 1

    accumulator_view = np.sum(accumulator, axis=0)

    return accumulator, accumulator_view


def n_max_ind(array: np.ndarray, n: int) -> np.ndarray:
    """Returns indices of n maximum values in an N-dimensional array.

    Args:
        array: Array
        n: Number of elements to returns. They are sorted in descending order.

    Returns:
        Indices of n max elements.
    """
    if n == 0:
        return np.asarray([])
    indices = array.ravel().argsort()[-n:]
    indices = np.unravel_index(indices, array.shape)
    return np.stack(indices, axis=1)


def general_hough(reference_image: np.ndarray, query_image: np.ndarray, path_for_saving: str,
                  threshold: float, is_line: bool) -> None:
    """Performs General Hough Transform.

    For any reference image and any query image, performs transform, marks locations and directions of
    the reference image on the query image, and saves results.

    Args:
        reference_image: The image containing a shape that is to be found on another image.
        query_image: The image that presumably contains that shape.
        path_for_saving: The path to the image that will contain results. By default, main method
            passes
        threshold: Controls how many points will be counted as centers. Lower value means more points.
            For line detection, consider 0.05 - 0.15. For other shapes, 0.85 - 0.95.
        is_line: Tells whether it is line that the algorithm should find.
            Flag is used for adjusting plot parameters.
    """
    r_table = build_r_table(reference_image)
    accumulator, accumulator_view = build_accumulator(r_table, query_image)

    plt.gray()
    fig, ax = plt.subplots(2, 2)
    ax1, ax2, ax3, ax4 = ax.ravel()
    ax1.set_title("Reference image")
    ax1.imshow(reference_image)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_title("Query image")
    ax2.imshow(query_image)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_title("Accumulator")
    ax3.imshow(accumulator_view)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_title("Detection")
    ax4.imshow(query_image)
    ax4.set_xticks([])
    ax4.set_yticks([])

    length = 100 if is_line else 10
    width = 2 if is_line else 0.5
    head_width = 0 if is_line else 2
    size = 0 if is_line else 10

    max_accum = np.max(accumulator)
    for theta, accum in enumerate(accumulator):
        greater = np.where(accum > threshold * max_accum, accum, 0)
        top = n_max_ind(greater, min(np.count_nonzero(greater), 4))
        for v_coord, h_coord in top:
            dx, dy = np.cos(theta * 2 * np.pi / THETA_NUMBERS), np.sin(
                theta * 2 * np.pi / THETA_NUMBERS
            )
            kwargs = {"width": width, "head_width": head_width, "color": "y"}
            ax4.arrow(h_coord, v_coord, -length * dy, -length * dx, **kwargs)
            ax4.scatter(h_coord, v_coord, marker='o', color='r', s=size)

    plt.tight_layout()
    plt.savefig(path_for_saving)


@click.command()
@click.option(
    "--reference_path",
    "-r",
    default="./images/line.png",
    help="Path to image with reference shape",
    show_default=True,
)
@click.option(
    "--query_path",
    "-q",
    default="./images/line_test5.png",
    help="Path to image that contains reference shape",
    show_default=True,
)
@click.option(
    "--angles",
    "-a",
    default=60,
    help="Number of angles to iterate through. Greater value means that more objects will be"
         "considered matching the shape. Must be less or equal than 360.",
    show_default=True,
)
@click.option(
    "--threshold",
    "-t",
    default=0.1,
    help="Controls how many points will be counted as centers. Lower value means more points."
         "For line detection, consider 0.05 - 0.15. For other shapes, 0.85 - 0.95.",
    show_default=True,
)
@click.option(
    "--is_line",
    "-l",
    is_flag=True,
    default=False,
    help="Option for showing that the reference image contains line. Used for adjusting the plot.",
    show_default=True,
)
@click.version_option(version=__version__)
def main(reference_path: str, query_path: str, angles: int, threshold: float, is_line: bool) -> None:
    """Main function for the general hough transform."""
    global THETA_NUMBERS
    THETA_NUMBERS = angles
    reference_image = imread(reference_path, as_gray=True)
    query_image = imread(query_path, as_gray=True)
    f = os.path.splitext(os.path.split(query_path)[1])[0]
    path_for_saving = os.path.join("./results/" + f + ".png")
    print(f"Image will be saved at {path_for_saving}")

    general_hough(reference_image, query_image, path_for_saving, threshold, is_line)

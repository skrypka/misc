import cv2
import PIL
import imageio
import numpy as np
from numba import jit

rgb_to_intensity_weights = np.array([0.2126, 0.7152, 0.0722])


def load_img(path):
    """
    Loads image, removes alpha channel and transfroms to float
    """
    return imageio.imread(path)[:, :, :3] / 255.


def save_img(path, img):
    return imageio.imwrite(path, (img * 255).astype(np.uint8))


def display_img(img):
    return PIL.Image.fromarray((img * 255).astype(np.uint8))


def rgb_to_intensity(img):
    """
    Convert image to grayscale (or one channel with intensity)
    """
    return np.einsum('ijl,l->ij', img, rgb_to_intensity_weights)


def normalize(array):
    array -= array.min()
    array /= array.max()
    return array


def slow_trilinear_interpolation(pos, grid):
    """
    x, y, z values from 0 to 1 (weight for every axes)
    grid is array with shape (2, 2, 2), if it's smaller
    we double it to correct dimmension

    https://en.wikipedia.org/wiki/Trilinear_interpolation
    https://en.wikipedia.org/wiki/Linear_interpolation
    """
    for ax in range(3):
        if grid.shape[ax] == 1:
            grid = np.concatenate((grid, grid), axis=ax)
    assert grid.shape == (2, 2, 2)

    x, y, z = pos
    for weight, axis in [(z, 2), (y, 1), (x, 0)]:
        # linear interpolation for `axis`
        grid = np.apply_along_axis(lambda line: line[0] * (1 - weight) + line[1] * weight,
                                   axis,
                                   grid)
    return grid


@jit
def trilinear_interpolation(pos, grid):
    """
    x, y, z values from 0 to 1 (weight for every axes)
    grid is array with shape (2, 2, 2), if it's smaller
    we double it to correct dimmension

    https://en.wikipedia.org/wiki/Trilinear_interpolation
    https://en.wikipedia.org/wiki/Linear_interpolation
    """
    for ax in range(3):
        if grid.shape[ax] == 1:
            grid = np.concatenate((grid, grid), axis=ax)
    assert grid.shape == (2, 2, 2)

    x, y, z = pos
    p00 = grid[0, 0, 0] * (1 - z) + grid[0, 0, 1] * z
    p01 = grid[0, 1, 0] * (1 - z) + grid[0, 1, 1] * z
    p10 = grid[1, 0, 0] * (1 - z) + grid[1, 0, 1] * z
    p11 = grid[1, 1, 0] * (1 - z) + grid[1, 1, 1] * z
    p0 = p00 * (1 - y) + p01 * y
    p1 = p10 * (1 - y) + p11 * y
    p = p0 * (1 - x) + p1 * x
    return p


def detect_edge(I, sigE, T, phiE):
    I1 = cv2.GaussianBlur(I, (5, 5), sigE)
    I2 = cv2.GaussianBlur(I, (5, 5), np.sqrt(1.6) * sigE)
    out = (I1 - T * I2)
    out_shape = out.shape
    for j in range(out_shape[1]):
        for i in range(out_shape[0]):
            if out[i, j] > 0:
                out[i, j] = 1
            elif out[i, j] <= 0:
                out[i, j] = 1 + np.tanh(phiE * out[i, j])
    return out

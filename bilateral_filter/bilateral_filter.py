import numpy as np
from scipy.ndimage import gaussian_filter

from utils import load_img, save_img, trilinear_interpolation, rgb_to_intensity, normalize


def create_grids(input_img, I, Sd, NbBuckets):
    grid_shape = (
        (input_img.shape[0] // Sd) + 2,
        (input_img.shape[1] // Sd) + 2,
        NbBuckets + 1,
    )
    grid_w = np.zeros(grid_shape)
    grid_r = np.zeros(grid_shape)
    grid_g = np.zeros(grid_shape)
    grid_b = np.zeros(grid_shape)

    # Grid creation
    for ix in range(input_img.shape[0]):
        for iy in range(input_img.shape[1]):
            grid_pos = (
                int(round(ix / Sd)),
                int(round(iy / Sd)),
                int(round(I[ix, iy])),
            )

            grid_w[grid_pos] += 1

            grid_r[grid_pos] += input_img[ix, iy, 0]
            grid_g[grid_pos] += input_img[ix, iy, 1]
            grid_b[grid_pos] += input_img[ix, iy, 2]

    return grid_w, grid_r, grid_g, grid_b


def process_grids_gaussian(grids, sigma):
    return [gaussian_filter(grid, sigma=sigma) for grid in grids]


# Slicing
def slicing(input_img, I, Sd, grids):
    result = np.zeros((input_img.shape[0], input_img.shape[1], 3))
    for ix in range(input_img.shape[0]):
        for iy in range(input_img.shape[1]):
            exact_pos = (ix / Sd, iy / Sd, I[ix, iy])
            grid_start_pos = [int(p) for p in exact_pos]

            grid_slice = tuple([slice(ax, ax + 2) for ax in grid_start_pos])
            rel_pos = tuple([p % 1 for p in exact_pos])

            near_area = [grid[grid_slice] for grid in grids]
            w, r, g, b = [trilinear_interpolation(rel_pos, area) for area in near_area]
            color = np.array([r, g, b]) / w
            result[ix, iy, :] = color
    return result


if __name__ == "__main__":
    Sd = 5  # dimension sample
    NbBuckets = 5  # intensity number

    input_img = load_img('img/before.png')
    # get intensity and normalize to [0, nb_buckets] range
    I = normalize(rgb_to_intensity(input_img)) * NbBuckets

    grids = create_grids(input_img, I, Sd, NbBuckets)
    grids = process_grids_gaussian(grids, sigma=0.5)
    result = slicing(input_img, I, Sd, grids)
    save_img("out.png", result)

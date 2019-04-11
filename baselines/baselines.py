import numpy as np
import scipy as sp


def bicubic(source_img, scaling_factor):
    source_img_size = source_img.shape[0]
    x_or_y = np.array(list(range(0, int(source_img_size)))).astype(float)
    int_img = sp.interpolate.RectBivariateSpline(x_or_y, x_or_y, source_img)
    x_or_y_up = np.array(list(range(0, source_img_size * scaling_factor))).astype(float) / scaling_factor - 0.5

    x_grid, y_grid = np.meshgrid(x_or_y_up, x_or_y_up, indexing="ij")
    return int_img.ev(x_grid, y_grid)
import os
import socket

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft

from make_jwst_kernels import get_pixscale

host = socket.gethostname()

if 'node' in host:
    base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
else:
    base_dir = '/Users/williams/Documents/phangs/jwst'

os.chdir(base_dir)

output_dir = 'convolved'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_filter = 'F770W'
output_filter = 'F2100W'
input_file = os.path.join('ngc7496',
                          'mastDownload',
                          'JWST',
                          'jw02107-o038_t019_miri_f770w',
                          'jw02107-o038_t019_miri_f770w_i2d.fits')
output_file = os.path.join(output_dir,
                           input_file.split(os.path.sep)[-1].replace('.fits', '_to_%s.fits' % output_filter))

kernel_hdu = fits.open(os.path.join('kernels', '%s_to_%s.fits' % (input_filter, output_filter)))[0]
input_hdu = fits.open(input_file)['SCI']

input_hdu.data[input_hdu.data == 0] = np.nan

kernel_pix_scale = get_pixscale(kernel_hdu)
image_pix_scale = get_pixscale(input_hdu)

# Interpolate kernel to same pixel grid as image

kernel_hdu_length = kernel_hdu.data.shape[0]

original_central_pixel = (kernel_hdu_length - 1) / 2

original_grid = (np.arange(kernel_hdu_length) - original_central_pixel) * kernel_pix_scale

# Calculate interpolated image size. Because sometimes there's a little pixel scale rounding error,
# subtract a little bit off the optimum size.

interpolate_image_size = np.floor(kernel_hdu_length * kernel_pix_scale / image_pix_scale) - 2

# Ensure the image has a central pixel

if interpolate_image_size % 2 == 0:
    interpolate_image_size -= 1

new_central_pixel = (interpolate_image_size - 1) / 2

new_grid = (np.arange(interpolate_image_size) - new_central_pixel) * image_pix_scale

x_coords_new, y_coords_new = np.meshgrid(new_grid, new_grid)

grid_interpolated = RegularGridInterpolator((original_grid, original_grid), kernel_hdu.data)
kernel_interp = grid_interpolated((x_coords_new.flatten(), y_coords_new.flatten()))
kernel_interp = kernel_interp.reshape(x_coords_new.shape)

input_hdu_convolved = convolve_fft(input_hdu.data, kernel_interp, allow_huge=True,
                                   preserve_nan=True, nan_treatment='interpolate')
input_hdu_convolved = fits.PrimaryHDU(input_hdu_convolved, input_hdu.header)

input_hdu_convolved.writeto(output_file, overwrite=True)

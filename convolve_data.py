import os
import socket
import glob

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

galaxies = ['ngc7496']

input_filters = ['F200W', 'F300M', 'F335M', 'F360M']
output_filters = ['F770W', 'F1000W', 'F1130W', 'F2100W']

for galaxy in galaxies:

    print('Starting %s' % galaxy)
    if not os.path.exists(os.path.join(output_dir, galaxy)):
        os.makedirs(os.path.join(output_dir, galaxy))

    for input_filter in input_filters:
        for output_filter in output_filters:

            # Find all relevant files
            input_files = glob.glob(os.path.join(galaxy.lower(),
                                                 'mastDownload',
                                                 'JWST',
                                                 '*%s' % input_filter.lower(),
                                                 '*_i2d.fits'))

            for input_file in input_files:

                output_fits_name = input_file.split(os.path.sep)[-1].replace('.fits', '_conv_to_%s.fits' % output_filter)
                output_file = os.path.join(output_dir, galaxy, output_fits_name)

                if os.path.exists(output_file):
                    continue

                print('Convolving %s to %s' % (input_filter, output_filter))

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

print('Complete!')

import itertools
import os
import socket

import numpy as np
from astropy.io import fits

from make_jwst_kernels import MakeConvolutionKernel, get_pixscale

host = socket.gethostname()

if 'node' in host:
    webbpsf_path = '/data/beegfs/astro-storage/groups/schinnerer/williams/webbpsf-data'
    base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data/kernels'
else:
    webbpsf_path = '/Users/williams/Documents/webbpsf-data'
    base_dir = '/Users/williams/Documents/phangs/jwst/kernels'

os.environ['WEBBPSF_PATH'] = webbpsf_path

import webbpsf

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

os.chdir(base_dir)

psf_folder = os.path.join('..', 'psfs')
if not os.path.exists(psf_folder):
    os.makedirs(psf_folder)

oversample_factor = 4
detector_oversample = 4
fov_arcsec = 20

overwrite = False

miri_filters = ['F770W', 'F1000W', 'F1130W', 'F2100W']
nircam_filters = ['F200W', 'F300M', 'F335M', 'F360M']
all_filters = nircam_filters + miri_filters

nircam = webbpsf.NIRCam()
miri = webbpsf.MIRI()

# filter_pairs = [['F770W', 'F2100W'],
#                 ['F770W', 'F1130W']]

filter_pairs = list(itertools.combinations(all_filters, 2))

for filter_pair in filter_pairs:
    input_filter = filter_pair[0]
    output_filter = filter_pair[1]

    # Check that the wavelengths work out, we can only go longer
    input_wavelength = int(input_filter[1:-1])
    output_wavelength = int(output_filter[1:-1])

    if output_wavelength <= input_wavelength:
        continue

    output_file = os.path.join('%s_to_%s.fits' % (input_filter, output_filter))

    if os.path.exists(output_file) and not overwrite:
        continue

    print('Generating kernel from %s -> %s' % (input_filter, output_filter))

    psfs = {}

    for jwst_filter in [input_filter, output_filter]:

        psf_name = os.path.join(psf_folder, '%s.fits' % jwst_filter)

        if not os.path.exists(psf_name) or overwrite:
            # Create PSFs. Use a detector_oversample of 1 so things are odd-shaped
            if jwst_filter in miri_filters:
                miri.filter = jwst_filter
                psf = miri.calc_psf(oversample=oversample_factor,
                                    detector_oversample=detector_oversample,
                                    # fov_pixels=fov_pixels,
                                    fov_arcsec=fov_arcsec,
                                    )[0]
            elif jwst_filter in nircam_filters:
                nircam.filter = jwst_filter
                psf = nircam.calc_psf(oversample=oversample_factor,
                                      detector_oversample=detector_oversample,
                                      # fov_pixels=fov_pixels,
                                      fov_arcsec=fov_arcsec,
                                      )[0]
            else:
                raise Warning('Not sure which camera filter %s uses' % jwst_filter)
            # print(psf.header)

            # Pad if we're awkwardly even
            if psf.data.shape[0] % 2 == 0:

                new_data = np.zeros([psf.data.shape[0] + 1, psf.data.shape[1] + 1])
                new_data[:-1, :-1] = psf.data
                psf.data = new_data

            psf.writeto(psf_name, overwrite=True)
        else:
            psf = fits.open(psf_name)[0]

        psfs[jwst_filter] = psf

    common_pixscale = get_pixscale(psfs[output_filter])

    grid_size_arcsec = np.array([3645 * common_pixscale,
                                 3645 * common_pixscale])

    conv_kern = MakeConvolutionKernel(source_psf=psfs[input_filter],
                                      source_name=input_filter,
                                      target_psf=psfs[output_filter],
                                      target_name=output_filter,
                                      grid_size_arcsec=grid_size_arcsec,
                                      common_pixscale=common_pixscale,
                                      verbose=False,
                                      )
    conv_kern.make_convolution_kernel()
    conv_kern.write_out_kernel()

print('Complete!')

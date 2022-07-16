import os
import socket
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils import make_source_mask
from reproject import reproject_interp


def sigma_clip(data, sigma=3, n_pixels=5, max_iterations=20):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = make_source_mask(data, nsigma=sigma, npixels=n_pixels)
        mean, median, std_dev = sigma_clipped_stats(data, mask=mask, sigma=sigma, maxiters=max_iterations)

    return [mean, median, std_dev]


def background_median(hdu, sigma=3, n_pixels=5, max_iterations=20):
    background = sigma_clip(hdu.data, sigma=sigma, n_pixels=n_pixels, max_iterations=max_iterations)[1]
    return background


host = socket.gethostname()

if 'node' in host:
    base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
else:
    base_dir = '/Users/williams/Documents/phangs/jwst'

os.chdir(base_dir)

galaxy = 'NGC7496'
nebula_mask = os.path.join('/Users/williams/Documents/phangs/nebulae_catalogue/DR2.0',
                           '%s_HIIreg_mask.fits' % galaxy)

file1 = os.path.join('convolved', 'jw02107-o038_t019_miri_f770w_i2d_to_F2100W.fits')
file2 = os.path.join('ngc7496',
                     'mastDownload',
                     'JWST',
                     'jw02107-o038_t019_miri_f2100w',
                     'jw02107-o038_t019_miri_f2100w_i2d.fits')

hdu1 = fits.open(file1)[0]
hdu2 = fits.open(file2)['SCI']
err_hdu = fits.open(file2)['ERR']

nebula_hdu = fits.open(nebula_mask)[0]
nebula_hdu.data[nebula_hdu.data > 0] = 1

hdu1.data[hdu1.data == 0] = np.nan
hdu2.data[hdu2.data == 0] = np.nan

hdu1.data -= background_median(hdu1)
hdu2.data -= background_median(hdu2)

mask = hdu2.data / err_hdu.data < 3

hdu1_reproj, _ = reproject_interp(hdu1, hdu2.header)
nebula_mask_reproj, _ = reproject_interp(nebula_hdu, hdu2.header, order='nearest-neighbor')

jj, ii = np.meshgrid(np.arange(nebula_mask_reproj.shape[1]),
                     np.arange(nebula_mask_reproj.shape[0]))

ratio = hdu1_reproj / hdu2.data

fits.writeto(os.path.join('convolved', 'F770W_F2100W_ratio.fits'),
             ratio, hdu2.header, overwrite=True)

ratio[mask] = np.nan
vmin, vmax = np.nanpercentile(ratio, [5, 95])

plt.figure(figsize=(8, 6))
ax = plt.subplot(111)
im = ax.imshow(ratio,
               vmin=vmin, vmax=vmax,
               cmap='inferno',
               origin='lower')
plt.contour(jj, ii, nebula_mask_reproj, 1, linewidths=1, colors=['green'])

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0)

plt.colorbar(im, cax=cax, label=r'$F_{770W}/F_{2100W}$')

plt.tight_layout()

plt.show()

print('Complete!')

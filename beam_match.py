import os
import socket

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate

host = socket.gethostname()

if 'node' in host:
    webbpsf_path = '/Users/williams/Documents/webbpsf-data'
else:
    webbpsf_path = '/Users/williams/Documents/webbpsf-data'

os.environ['WEBBPSF_PATH'] = webbpsf_path

import webbpsf

nc = webbpsf.NIRCam()
nc.filter = 'F200W'
psf = nc.calc_psf(oversample=2)[0]

vmin, vmax = np.nanpercentile(psf.data, [1, 99])

plt.figure()
plt.imshow(psf.data, origin='lower', vmin=vmin, vmax=vmax)

plt.title('Orig')
plt.axis('off')

plt.tight_layout()

plt.show()

print('Complete!')

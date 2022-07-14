import os
import socket
import warnings
from datetime import datetime

from astroquery.exceptions import NoResultsWarning
from astroquery.mast import Observations
import numpy as np


def get_time():
    """Get current time as a string"""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    return current_time


warnings.simplefilter('error', NoResultsWarning)

host = socket.gethostname()

if 'node' in host:
    dl_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
else:
    dl_dir = '/Users/williams/Documents/phangs/jwst'

os.chdir(dl_dir)

instrument = 'JWST'
prop_id = '02107'
calib_level = [2, 3]
extension = 'fits'  # None after initial run?
jwst_instrument = 'NIRCAM/IMAGE'

targets = [
    'NGC-7496',
    # 'IC-5332',
    # 'NGC-628',
]

print('[%s] Downloading from proposal %s with calib_level %s, extension %s' %
      (get_time(), prop_id, calib_level, extension))

do_filter = True
for target in targets:

    print('[%s] Starting download for %s' % (get_time(), target))

    obs_list = Observations.query_criteria(obs_collection=instrument,
                                           proposal_id=prop_id,
                                           target_name=target)

    if np.all(obs_list['calib_level'] == -1):
        print('[%s] No available data for %s' % (get_time(), target))
        # continue

    if jwst_instrument is not None:
        obs_list = obs_list[obs_list['instrument_name'] == jwst_instrument]

    for obs in obs_list:
        try:
            product_list = Observations.get_product_list(obs)
        except NoResultsWarning:
            print('[%s] Data not available for %s' % (get_time(), obs['obs_id']))
            continue

        print('[%s] Downloading %s' % (get_time, obs['obs_id']))

        if do_filter:
            products = Observations.filter_products(product_list,
                                                    calib_level=calib_level,
                                                    extension=extension)
            if len(products) > 0:
                manifest = Observations.download_products(products)
        else:
            manifest = Observations.download_products(product_list)

print('Complete!')

import glob
import os
import socket
import warnings
from datetime import datetime

import numpy as np
from astroquery.exceptions import NoResultsWarning
from astroquery.mast import Observations

from api_key import api_key


def get_time():
    """Get current time as a string"""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    return current_time


warnings.simplefilter('error', NoResultsWarning)

host = socket.gethostname()

if 'node' in host:
    base_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_data'
else:
    base_dir = '/Users/williams/Documents/phangs/jwst'

os.chdir(base_dir)

observations = Observations()

instrument = 'JWST'
prop_id = '2107'

calib_level = [
    # 1,
    2,
    3,
]

extension = None
instrument_name = None
login = True

if login:
    observations.login(token=api_key)

targets = [
    # 'ngc7496',
    # 'ic5332',
    'ngc0628',
]

product_type = [
    'SCIENCE',
    'PREVIEW',
    'INFO',
    # 'AUXILIARY',
]

print('[%s] Downloading from proposal %s with calib_level %s, extension %s, instrument %s' %
      (get_time(), prop_id, calib_level, extension, instrument_name))

do_filter = True
for target in targets:

    print('[%s] Starting download for %s' % (get_time(), target))

    obs_list = Observations.query_object(target)

    if np.all(obs_list['calib_level'] == -1):
        print('[%s] No available data for %s' % (get_time(), target))
        continue

    obs_list = obs_list[obs_list['obs_collection'] == instrument]
    obs_list = obs_list[obs_list['proposal_id'] == prop_id]
    obs_list = obs_list[obs_list['calib_level'] >= 0]

    # obs_list = observations.query_criteria(
    #     obs_collection=instrument,
    #     proposal_id=prop_id,
    #     target_name=target,
    # )

    dl_dir = target.replace(' ', '_')
    if not os.path.exists(dl_dir):
        os.makedirs(dl_dir)
    os.chdir(dl_dir)

    if instrument_name is not None:
        obs_list = obs_list[obs_list['instrument_name'] == instrument_name]

    for obs in obs_list:
        try:
            product_list = observations.get_product_list(obs)
        except NoResultsWarning:
            print('[%s] Data not available for %s' % (get_time(), obs['obs_id']))
            continue

        print('[%s] Downloading %s' % (get_time(), obs['obs_id']))

        if do_filter:
            # print(list(np.unique(product_list['productType'])))
            products = observations.filter_products(product_list,
                                                    calib_level=calib_level,
                                                    productType=product_type,
                                                    extension=extension)
            if len(products) > 0:
                manifest = observations.download_products(products)
            else:
                print('[%s] Filtered data not available' % get_time())
        else:
            manifest = observations.download_products(product_list)

    os.chdir(base_dir)

print('Complete!')

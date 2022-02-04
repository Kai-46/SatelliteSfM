import os
import json
import argparse
from preprocess.preprocess_image_set import preprocess_image_set
import srtm4
import numpy as np
from icecream import ic


parser = argparse.ArgumentParser()

parser.add_argument('--input_folder', type=str, default=None, help='Folder containing input data.')
parser.add_argument('--output_folder', type=str, default=None, help='Folder for storing output files.')
parser.add_argument('--use_srtm4', action='store_true', default=False, help='Whether to guess altitude range based on SRTM4 elevation data.')
parser.add_argument('--enable_debug', action='store_true', default=False, help='Whether to enable debug mode.')
parser.add_argument('--run_sfm', action='store_true', default=False, help='Whether to run structure from motion.')
args = parser.parse_args()
ic(args)

if __name__ == '__main__':
    latlonalt_bbx = json.load(open(os.path.join(args.input_folder, 'latlonalt_bbx.json')))

    if args.use_srtm4:
        altitude = srtm4.srtm4(np.mean(latlonalt_bbx['lon_minmax']),
                           np.mean(latlonalt_bbx['lat_minmax']))
        latlonalt_bbx['alt_minmax'] = [altitude - 10, 300]
        ic('altitude range from SRTM4: ', latlonalt['alt_minmax'])

    preprocess_image_set(args.output_folder, os.path.join(args.input_folder, 'images'),
                         latlonalt_bbx['lat_minmax'], latlonalt_bbx['lon_minmax'], latlonalt_bbx['alt_minmax'],
                         enable_debug=args.enable_debug, run_sfm=args.run_sfm)

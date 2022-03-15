import sys
sys.path.append('../')


import numpy as np
import json
import os
import pyproj
import pymap3d
import tifffile
import numpy_groupies as npg
import cv2
import pyexr
import utm
import multiprocessing
import imageio


from preprocess.rpc_model import RPCModel
from preprocess.approximate_rpc_locally import approximate_rpc_locally
from preprocess.parse_tif_image import parse_tif_image

from icecream import ic


def latlon_to_eastnorh(lat, lon):
    # assume all the points are either on north or south hemisphere
    assert(np.all(lat >= 0) or np.all(lat < 0))

    if lat[0, 0] >= 0: # north hemisphere
        south = False
    else:
        south = True

    _, _, zone_number, _ = utm.from_latlon(lat[0, 0], lon[0, 0])

    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=south)
    east, north = proj(lon, lat)
    return east, north

def eastnorth_to_latlon(east, north, zone_number, hemisphere):
    if hemisphere == 'N':
        south = False
    else:
        south = True

    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=south)
    lon, lat = proj(east, north, inverse=True)
    return lat, lon

def latlonalt_to_enu(lat, lon, alt, lat0, lon0, alt0):
    e, n, u = pymap3d.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)

    return e, n, u

def enu_to_latlonalt(e, n, u, lat0, lon0, alt0):
    lat, lon, alt = pymap3d.enu2geodetic(e, n, u, lat0, lon0, alt0)

    return lat, lon, alt


def single_worker(out_dir, dsm_tif_fpath, view_tif_fpath):
    ic(out_dir, dsm_tif_fpath, view_tif_fpath)
    # read tile bounds
    easting, northing, pixels, gsd = np.loadtxt(dsm_tif_fpath[:-4]+'.txt')   # lower-left corner
    pixels = int(pixels)
    ul_utm_e = easting
    ul_utm_n = northing + (pixels - 1) * gsd
    site_width = pixels * gsd
    site_height = pixels * gsd

    # read dsm
    dsm = tifffile.imread(dsm_tif_fpath)

    # determine observer location for local ENU
    alt_min = float(np.nanmin(dsm))
    alt_max = float(np.nanmax(dsm))
    observ_utm_e = easting + (pixels - 1) / 2. * gsd
    observ_utm_n = northing + (pixels - 1) / 2. * gsd
    if 'JAX' in os.path.basename(dsm_tif_fpath):
        zone_number = 17
        hemisphere = 'N'
    elif 'OMA' in os.path.basename(dsm_tif_fpath):
        zone_number = 15
        hemisphere = 'N'
    observ_lat, observ_lon = eastnorth_to_latlon(observ_utm_e, observ_utm_n, zone_number, hemisphere=hemisphere)
    observ_alt = alt_min - 10
    # ic('observer: ', observ_lat, observ_lon, observ_alt)

    # lift dsm to 3d point cloud
    dsm = cv2.resize(dsm, (2*dsm.shape[1], 2*dsm.shape[0]), interpolation=cv2.INTER_NEAREST)   # densify lifted point cloud
    utm_e, utm_n = np.meshgrid(np.linspace(ul_utm_e, ul_utm_e+site_width, dsm.shape[1]),
                               np.linspace(ul_utm_n, ul_utm_n-site_height, dsm.shape[0]))
    utm_e = utm_e.reshape((-1))
    utm_n = utm_n.reshape((-1))
    dsm = dsm.reshape((-1))

    lat, lon = eastnorth_to_latlon(utm_e, utm_n, zone_number=zone_number, hemisphere=hemisphere)
    enu_e, enu_n, enu_u = latlonalt_to_enu(lat, lon, dsm, observ_lat, observ_lon, observ_alt)

    ############ project to image grid
    # read input tif image
    img, meta = parse_tif_image(view_tif_fpath)
    rpc_model = RPCModel(meta)
    col, row = rpc_model.projection(lat, lon, dsm)
    xsize, ysize = meta['width'], meta['height']
    col, row = np.round(col).astype(int), np.round(row).astype(int)
    points_group_idx = row * xsize + col
    points_val = enu_u   # in the image grid, each pixel stores the enu up value

    # remove points that lie out of the dsm boundary
    mask = ((row >= 0) * (col >= 0) * (row < ysize) * (col < xsize)) > 0
    points_group_idx = points_group_idx[mask]
    points_val = points_val[mask]

    # create a place holder for all pixels in the dsm
    group_idx = np.arange(xsize * ysize).astype(dtype=int)
    group_val = np.empty(xsize * ysize)
    group_val.fill(np.nan)

    # concatenate place holders with the real valuies, then aggregate
    group_idx = np.concatenate((group_idx, points_group_idx))
    group_val = np.concatenate((group_val, points_val))

    dsm = npg.aggregate(group_idx, group_val, func='nanmax', fill_value=np.nan)
    dsm = dsm.reshape((ysize, xsize))

    # try to fill very small holes
    dsm_new = dsm.copy()
    nan_places = np.argwhere(np.isnan(dsm_new))
    for i in range(nan_places.shape[0]):
        row = nan_places[i, 0]
        col = nan_places[i, 1]
        neighbors = []
        for j in range(row-1, row+2):
            for k in range(col-1, col+2):
                if j >= 0 and j < dsm_new.shape[0] and k >=0 and k < dsm_new.shape[1]:
                    val = dsm_new[j, k]
                    if not np.isnan(val):
                        neighbors.append(val)

        if neighbors:
            dsm[row, col] = np.median(neighbors)

    dsm[np.isnan(dsm)] = 0.
    dsm = dsm.astype(np.float32)
    ############ end of project to image grid

    # derive approx
    lat_minmax = [np.nanmin(lat), np.nanmax(lat)]
    lon_minmax = [np.nanmin(lon), np.nanmax(lon)]
    alt_minmax = [alt_min-5., alt_max+5.]
    K, W2C = approximate_rpc_locally(meta, lat_minmax, lon_minmax, alt_minmax,
                                        observ_lat, observ_lon, observ_alt)
    cam_dict = {
        'K': K.flatten().tolist(),
        'W2C': W2C.flatten().tolist(),
        'img_size': [img.shape[1], img.shape[0]]
    }

    # output to multiple subdirectories
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'metas'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'cameras'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'enu_observers'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'enu_bbx'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'groundtruth_u'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'latlonalt_bbx'), exist_ok=True)

    base_name = os.path.basename(view_tif_fpath)
    imageio.imwrite(os.path.join(out_dir, 'images', base_name[:-4]+'.png'), img)
    with open(os.path.join(out_dir, 'metas', base_name[:-4]+'.json'), 'w') as fp:
        json.dump(meta, fp, indent=2)
    with open(os.path.join(out_dir, 'enu_observers', base_name[:-4]+'.json'), 'w') as fp:
        json.dump([observ_lat, observ_lon, observ_alt], fp)
    with open(os.path.join(out_dir, 'cameras', base_name[:-4]+'.json'), 'w') as fp:
        json.dump(cam_dict, fp, indent=2)
    enu_bbx = {
        'e_minmax': [np.nanmin(enu_e), np.nanmax(enu_e)],
        'n_minmax': [np.nanmin(enu_n), np.nanmax(enu_n)],
        'u_minmax': [np.nanmin(enu_u)-5., np.nanmax(enu_u)+5.] }
    with open(os.path.join(out_dir, 'enu_bbx', base_name[:-4]+'.json'), 'w') as fp:
        json.dump(enu_bbx, fp, indent=2)
    latlonalt_bbx = {
        'lat_minmax': lat_minmax,
        'lon_minmax': lon_minmax,
        'alt_minmax': alt_minmax
    }
    with open(os.path.join(out_dir, 'latlonalt_bbx', base_name[:-4]+'.json'), 'w') as fp:
        json.dump(latlonalt_bbx, fp, indent=2)
    pyexr.write(os.path.join(out_dir, 'groundtruth_u', base_name[:-4]+'.exr'), dsm)


def multiple_workers(out_dir, dsm_tif_fpath_list, view_tif_fpath_list, max_processes=-1):
    if max_processes <= 0:
        max_processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(max_processes)
    for dsm_tif_fpath, view_tif_fpath in zip(dsm_tif_fpath_list, view_tif_fpath_list):
        pool.apply_async(single_worker, args=(out_dir, dsm_tif_fpath, view_tif_fpath))

    pool.close()
    pool.join()


if __name__ == '__main__':
    # dsm_tif_fpath = '/phoenix/S7/IARPA-SMART/delivery/train_mvs/Track3-Truth/JAX_004_DSM.tif'
    # view_tif_fpath = '/phoenix/S7/IARPA-SMART/delivery/train_mvs/Track3-Train/JAX_004_006_RGB.tif'
    # out_dir = '/phoenix/S7/kz298/dfc2019_preprocessed'
    # single_worker(out_dir, dsm_tif_fpath, view_tif_fpath)

    base_view_dir = '/phoenix/S7/IARPA-SMART/delivery/train_mvs/Track3-Train'
    base_dsm_dir = '/phoenix/S7/IARPA-SMART/delivery/train_mvs/Track3-Truth'
    out_dir = '/phoenix/S7/kz298/dfc2019_preprocessed'
    views = [x for x in os.listdir(base_view_dir) if x.endswith('.tif')]
    dsms = [x[:7]+'_DSM.tif' for x in views]

    dsm_tif_fpath_list = [os.path.join(base_dsm_dir, x) for x in dsms]
    view_tif_fpath_list = [os.path.join(base_view_dir, x) for x in views]
    ic(len(dsm_tif_fpath_list), len(view_tif_fpath_list))
    ic(dsm_tif_fpath_list[:5])
    ic(view_tif_fpath_list[:5])
    # exit(0)
    for x in dsm_tif_fpath_list:
        assert(os.path.isfile(x)),f'{x}'
    for x in view_tif_fpath_list:
        assert(os.path.isfile(x)),f'{x}'

    multiple_workers(out_dir, dsm_tif_fpath_list, view_tif_fpath_list)

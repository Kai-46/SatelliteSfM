import itertools
from logging import debug
import os
import numpy as np
import json
import imageio
import multiprocessing
from icecream import ic
import datetime

from preprocess.approximate_rpc_locally import approximate_rpc_locally
from preprocess.parse_tif_image import parse_tif_image
from preprocess.visual_inspect_camera import warp_src_to_ref
from preprocess.coordinate_system import latlonalt_to_enu

from preprocess_sfm.preprocess_sfm import preprocess_sfm

rmext = lambda x: x[0:x.rfind('.')]


def _pad_images_to_samesize(in_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    all_img_names = []
    all_imgs = []
    for item in sorted(os.listdir(in_folder)):
        if item.endswith('.png'):
            all_img_names.append(item)
            im = imageio.imread(os.path.join(in_folder, item))
            if len(im.shape) == 2:
                im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
            all_imgs.append(im)

    all_H = [im.shape[0] for im in all_imgs]
    all_W = [im.shape[1] for im in all_imgs]
    trgt_H = np.max(all_H)
    trgt_W = np.max(all_W)

    trgt_H = int(np.ceil(trgt_H / 16) * 16)
    trgt_W = int(np.ceil(trgt_W / 16) * 16)

    for im, im_name in zip(all_imgs, all_img_names):
        pad = [(0, trgt_H - im.shape[0]), (0, trgt_W - im.shape[1]), (0, 0)]
        im_pad = np.pad(im, pad)
        imageio.imwrite(os.path.join(out_folder, im_name), im_pad)

    return trgt_H, trgt_W


def _preprocess_single_tif(args):
    out_folder, tif_image_folder, item, lat_minmax, lon_minmax, alt_minmax, observer_lat, observer_lon, observer_alt = args

    img, meta_dict = parse_tif_image(os.path.join(tif_image_folder, item))
    # ic(img.shape, type(img), img.dtype)
    imageio.imwrite(os.path.join(out_folder, 'images', rmext(item)+'.png'), img)
    with open(os.path.join(out_folder, 'metas', rmext(item)+'.json'), 'w') as fp:
        json.dump(meta_dict, fp, indent=2)
    K, W2C = approximate_rpc_locally(meta_dict, lat_minmax, lon_minmax, alt_minmax,
                                                observer_lat, observer_lon, observer_alt)

    cam_dict = {
        'K': K.flatten().tolist(),
        'W2C': W2C.flatten().tolist(),
        'img_size': [img.shape[1], img.shape[0]]
    }
    with open(os.path.join(out_folder, 'cameras', rmext(item)+'.json'), 'w') as fp:
        json.dump(cam_dict, fp, indent=2)


def preprocess_image_set(out_folder, tif_image_folder, lat_minmax, lon_minmax, alt_minmax,
                         enable_debug=False, run_sfm=False):
    '''
        tif_image_folder: folder containing tif images for the site; see ../examples/dfc_data/inputs

        lat_minmax, lon_minmax, alt_minmax: (2, ); bounding box of the site of interest
    '''
    # convert tif to png; extract tif header; and approximate rpc with pinhole
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'metas'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'cameras'), exist_ok=True)

    latlonalt_bbx = {
        'lat_minmax': lat_minmax,
        'lon_minmax': lon_minmax,
        'alt_minmax': alt_minmax
    }
    with open(os.path.join(out_folder, 'latlonalt_bbx.json'), 'w') as fp:
        json.dump(latlonalt_bbx, fp, indent=2)
    ic(latlonalt_bbx)

    observer_lat = (lat_minmax[0] + lat_minmax[1]) / 2.
    observer_lon = (lon_minmax[0] + lon_minmax[1]) / 2.
    observer_alt = np.min(alt_minmax) - 20.   # 20 meter is a margin for numeric stability
    with open(os.path.join(out_folder, 'enu_observer_latlonalt.json'), 'w') as fp:
        json.dump([observer_lat, observer_lon, observer_alt], fp)
    ic(observer_lat, observer_lon, observer_alt)

    latlonalt_pts = np.array(list(itertools.product(list(lat_minmax), list(lon_minmax), list(alt_minmax))))
    # ic(latlonalt_pts.shape)
    e, n, u = latlonalt_to_enu(latlonalt_pts[:, 0], latlonalt_pts[:, 1], latlonalt_pts[:, 2],
                               observer_lat, observer_lon, observer_alt)
    enu_bbx = {
        'e_minmax': [np.min(e), np.max(e)],
        'n_minmax': [np.min(n), np.max(n)],
        'u_minmax': [np.min(u)-10., np.max(u)+10.] }
    with open(os.path.join(out_folder, 'enu_bbx.json'), 'w') as fp:
        json.dump(enu_bbx, fp, indent=2)
    ic(enu_bbx)

    all_tif_items = [x for x in os.listdir(tif_image_folder) if x.endswith('.tif')]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    job_args = [(out_folder, tif_image_folder, x, lat_minmax, lon_minmax, alt_minmax, observer_lat, observer_lon, observer_alt) for x in all_tif_items]
    pool.map(_preprocess_single_tif, job_args)
    pool.close()
    pool.join()

    # # pad images to same size
    # padded_height, padded_width = _pad_images_to_samesize(os.path.join(out_folder, 'images'), os.path.join(out_folder, 'padded_images'))
    # os.makedirs(os.path.join(out_folder, 'padded_cameras'), exist_ok=True)
    # for item in os.listdir(os.path.join(out_folder, 'cameras')):
    #     with open(os.path.join(out_folder, 'cameras', item)) as fp:
    #         cam_dict = json.load(fp)
    #     cam_dict['img_size'] = [padded_width, padded_height]
    #     with open(os.path.join(out_folder, 'padded_cameras', item), 'w') as fp:
    #         json.dump(cam_dict, fp, indent=2)

    # run sfm to bundle-adjust cameras
    # take inputs from {out_folder}/images, {out_folder}/cameras
    #      and output to {out_folder}/cameras_adjusted, {out_folder}/enu_bbx_adjusted.json, and {out_folder}/debug_sfm
    if run_sfm:
        preprocess_sfm(out_folder)

    # warp src to ref for debugging purpose
    if enable_debug:
        if run_sfm:
            debug_folder = os.path.join(out_folder, 'debug_sfm')
            camera_dir = os.path.join(out_folder, 'cameras_adjusted')
            enu_bbx = json.load(open(os.path.join(out_folder, 'enu_bbx_adjusted.json')))
        else:
            debug_folder = os.path.join(out_folder, 'debug_preprocess')
            camera_dir = os.path.join(out_folder, 'cameras')
            enu_bbx = json.load(open(os.path.join(out_folder, 'enu_bbx.json')))

        ic(debug_folder)
        os.makedirs(debug_folder, exist_ok=True)
        img_names = [x for x in os.listdir(os.path.join(out_folder, 'images')) if x.endswith('.png')]
        debug_ref_img_names = np.random.choice(img_names, 2, replace=False)
        for ref_img_name in debug_ref_img_names:
            while True:
                src_img_name = np.random.choice(img_names, 1, replace=False)[0]
                if src_img_name != ref_img_name: break
            # ic(ref_img_name, src_img_name)
            tmp_out_dir = os.path.join(debug_folder, rmext(ref_img_name)+','+rmext(src_img_name))
            ref_im = imageio.imread(os.path.join(out_folder, 'images', ref_img_name))
            src_im = imageio.imread(os.path.join(out_folder, 'images', src_img_name))
            ref_cam_dict = json.load(open(os.path.join(camera_dir, rmext(ref_img_name)+'.json')))
            src_cam_dict = json.load(open(os.path.join(camera_dir, rmext(src_img_name)+'.json')))
            ref_P = np.array(ref_cam_dict['K']).reshape((4, 4)) @ np.array(ref_cam_dict['W2C']).reshape((4, 4))
            src_P = np.array(src_cam_dict['K']).reshape((4, 4)) @ np.array(src_cam_dict['W2C']).reshape((4, 4))
            num_planes = int((enu_bbx['u_minmax'][1] - enu_bbx['u_minmax'][0]) / 0.2)
            sweep_plane_sequence = np.zeros((num_planes, 4))
            sweep_plane_sequence[:, 2] = 1
            sweep_plane_sequence[:, 3] = np.linspace(enu_bbx['u_minmax'][0], enu_bbx['u_minmax'][1], num_planes)
            warp_src_to_ref(tmp_out_dir, sweep_plane_sequence, ref_im, src_im, ref_P, src_P, subarea=None, max_processes=None)



if __name__ == '__main__':
    latlonalt_bbx = json.load(open('../examples/inputs/latlonalt_bbx.json'))
    lat_minmax = latlonalt_bbx['lat_minmax']
    lon_minmax = latlonalt_bbx['lon_minmax']
    alt_minmax = latlonalt_bbx['alt_minmax']

    tif_image_folder = '../examples/inputs/images'
    out_folder = '../examples/outputs'

    preprocess_image_set(out_folder, tif_image_folder, lat_minmax, lon_minmax, alt_minmax)

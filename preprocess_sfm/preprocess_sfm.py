import os
import json
import trimesh
import numpy as np

import preprocess_sfm.colmap_sfm_commands as colmap_sfm_commands
from preprocess_sfm.colmap_sfm_utils import extract_camera_dict, extract_all_to_dir


rmext = lambda x: x[0:x.rfind('.')]


def preprocess_sfm(out_folder, weight=0.01):
    debug_folder = os.path.join(out_folder, 'debug_sfm')
    os.makedirs(debug_folder, exist_ok=True)

    # run sift matching
    img_dir = os.path.join(out_folder, 'images')
    db_file = os.path.join(out_folder, 'database.db')
    colmap_sfm_commands.run_sift_matching(img_dir, db_file,
                                          debug_folder=debug_folder)
    
    # prepare initial guss of camera parameters
    all_img_names = [x for x in os.listdir(img_dir) if x.endswith('.png')]
    cam_dict = {}
    for x in all_img_names:
        cam_dict[x] = json.load(open(os.path.join(out_folder, 'cameras', rmext(x)+'.json')))
    
    cam_dict_file = os.path.join(out_folder, 'cam_dict.json')
    with open(cam_dict_file, 'w') as fp:
        json.dump(cam_dict, fp, indent=2)

    # iterate between triangulation and bundle adjustment
    cam_dict_adjusted_file = os.path.join(out_folder, 'cam_dict_adjusted.json')
    os.system('cp {} {}'.format(cam_dict_file, 
                                cam_dict_adjusted_file))
    for reproj_err_threshold in [32.0, 2.0]:
        # triangulate
        tri_dir = os.path.join(out_folder, 'colmap_triangulate')
        colmap_sfm_commands.run_point_triangulation(img_dir, db_file, tri_dir, cam_dict_adjusted_file,
                                                    reproj_err_threshold, reproj_err_threshold, reproj_err_threshold,
                                                    debug_folder=debug_folder)

        # global bundle adjustment
        tri_ba_dir = os.path.join(out_folder, 'colmap_triangulate_postba')
        colmap_sfm_commands.run_global_ba(tri_dir, tri_ba_dir, weight, debug_folder=debug_folder)

        # update camera dict
        cam_dict_adjusted = extract_camera_dict(tri_ba_dir)
        with open(cam_dict_adjusted_file, 'w') as fp:
            json.dump(cam_dict_adjusted, fp, indent=2)

    extract_all_to_dir(tri_ba_dir, os.path.join(out_folder, 'debug_sfm'))
    
    # write adjusted cameras
    os.makedirs(os.path.join(out_folder, 'cameras_adjusted'), exist_ok=True)
    for img_name, cam in cam_dict_adjusted.items():
        with open(os.path.join(out_folder, 'cameras_adjusted', rmext(img_name)+'.json'), 'w') as fp:
            json.dump(cam, fp, indent=2)

    # write adjusted enu bounding box
    points = trimesh.load(os.path.join(out_folder, 'debug_sfm/kai_points.ply')).vertices
    z_min, z_max = np.percentile(points[:, 2], (2, 98))
    z_min -= 20  # add margin to make sure scene is fully bounded
    z_max += 20
    
    enu_bbx = json.load(open(os.path.join(out_folder, 'enu_bbx.json')))
    enu_bbx['u_minmax'] = [z_min, z_max]
    with open(os.path.join(out_folder, 'enu_bbx_adjusted.json'), 'w') as fp:
        json.dump(enu_bbx, fp, indent=2)

    # check how big the image-space translations are
    with open(os.path.join(out_folder, 'cam_dict.json')) as fp:
        preba_cameras = json.load(fp)
    with open(os.path.join(out_folder, 'cam_dict_adjusted.json')) as fp:
        postba_cameras = json.load(fp)

    result = ['img_name, delta_cx, delta_cy\n', ]
    for img_name in sorted(preba_cameras.keys()):
        preba = preba_cameras[img_name]
        postba = postba_cameras[img_name]
        delta_cx = postba['K'][2] - preba['K'][2]
        delta_cy = postba['K'][6] - preba['K'][6]
        result.append('{}, {}, {}\n'.format(img_name, delta_cx, delta_cy))

    with open(os.path.join(out_folder, 'principal_points_adjustment.csv'), 'w') as fp:
        fp.write(''.join(result))

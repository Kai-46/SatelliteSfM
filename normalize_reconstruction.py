import os
import numpy as np
import json
import shutil

#import sys
#sfm_dir = sys.argv[1]

sfm_dir = './examples/outputs'
out_dir = './examples/outputs_normalized'

os.makedirs(out_dir, exist_ok=True)
shutil.copytree(os.path.join(sfm_dir, 'images'),
                os.path.join(out_dir, 'images'))
os.makedirs(os.path.join(out_dir, 'cameras'), exist_ok=True)


camera_dir = os.path.join(sfm_dir, 'cameras_adjusted')
enu_bbx_adjusted = json.load(open(os.path.join(sfm_dir, 'enu_bbx_adjusted.json')))

scene_center = [np.mean(enu_bbx_adjusted['e_minmax']),
                np.mean(enu_bbx_adjusted['n_minmax']),
                np.mean(enu_bbx_adjusted['u_minmax'])]
scene_scale = np.sqrt(
    (np.max(enu_bbx_adjusted['e_minmax']) - np.min(enu_bbx_adjusted['e_minmax'])) ** 2 +  \
    (np.max(enu_bbx_adjusted['n_minmax']) - np.min(enu_bbx_adjusted['n_minmax'])) ** 2 +  \
    (np.max(enu_bbx_adjusted['u_minmax']) - np.min(enu_bbx_adjusted['u_minmax'])) ** 2
)

# shift scene center to origin, and scale scene to target sphere
def transform_pose(W2C, translate, scale):
    C2W = np.linalg.inv(W2C)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    return np.linalg.inv(C2W)

translate = -np.array(scene_center)
trgt_sphere_radius = 0.75
scale = trgt_sphere_radius * 2. / scene_scale

for item in os.listdir(camera_dir):
    cam_dict = json.load(open(os.path.join(camera_dir, item)))
    W2C = np.array(cam_dict['W2C']).reshape((4, 4))
    W2C = transform_pose(W2C, translate, scale)
    cam_dict = {
        'img_size': cam_dict['img_size'],
        'K': cam_dict['K'],
        'W2C': W2C.flatten().tolist()
    }

    with open(os.path.join(out_dir, 'cameras', item), 'w') as fp:
        json.dump(cam_dict, fp, indent=2, sort_keys=True)


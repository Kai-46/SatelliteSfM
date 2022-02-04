import numpy as np
import json
import os
from pyquaternion import Quaternion
import trimesh

from preprocess_sfm.colmap.read_model import read_model
import preprocess_sfm.colmap.database as database


def init_posed_sfm(db_file, cam_dict_file, out_dir):
    '''
    create cameras.txt, images.txt, points3D.txt from existing camera parameters
    make sure cameras are numbered in the same way as db_file
    '''
    # read database and get the mapping from image name to image id
    db = database.COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    # create files required by colmap sfm
    cam_dict = json.load(open(cam_dict_file))

    cameras_txt_lines = []
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        cameras_line_template = '{camera_id} PERSPECTIVE {width} {height} {fx} {fy} {cx} {cy} {s}\n'
        images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

        K = np.array(cam_dict[img_name]['K']).reshape((4, 4))
        W2C = np.array(cam_dict[img_name]['W2C']).reshape((4, 4))
        width, height = cam_dict[img_name]['img_size']
        quat = Quaternion(matrix=W2C[:3, :3])
    
        cameras_line = cameras_line_template.format(
            camera_id=img_id, width=width, height=height, fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2], s=K[0,1]
        )         
        images_line = images_line_template.format(
            image_id=img_id, qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3], tx=W2C[0,3], ty=W2C[1,3], tz=W2C[2,3],
            camera_id=img_id, image_name=img_name
        )
        
        cameras_txt_lines.append(cameras_line)
        images_txt_lines.append(images_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()


def _read_tracks(colmap_images, colmap_points3D):
    all_tracks = []     # list of dicts; each dict represents a track
    all_points = []     # list of all 3D points
    view_keypoints = {} # dict of lists; each list represents the triangulated key points of a view

    for point3D_id in colmap_points3D:
        point3D = colmap_points3D[point3D_id]
        image_ids = point3D.image_ids
        point2D_idxs = point3D.point2D_idxs

        cur_track = {}
        cur_track['xyz'] = (point3D.xyz[0], point3D.xyz[1], point3D.xyz[2])
        cur_track['err'] = point3D.error

        cur_track_len = len(image_ids)
        assert (cur_track_len == len(point2D_idxs))
        all_points.append(list(cur_track['xyz'] + (cur_track['err'], cur_track_len) + tuple(point3D.rgb)))

        pixels = []
        for i in range(cur_track_len):
            image = colmap_images[image_ids[i]]
            img_name = image.name
            point2D_idx = point2D_idxs[i]
            point2D = image.xys[point2D_idx]
            assert (image.point3D_ids[point2D_idx] == point3D_id)
            pixels.append((img_name, point2D[0], point2D[1]))

            if img_name not in view_keypoints:
                view_keypoints[img_name] = [(point2D[0], point2D[1]) + cur_track['xyz'] + (cur_track_len, ), ]
            else:
                view_keypoints[img_name].append((point2D[0], point2D[1]) + cur_track['xyz'] + (cur_track_len, ))

        cur_track['pixels'] = sorted(pixels, key=lambda x: x[0]) # sort pixels by the img_name
        all_tracks.append(cur_track)

    return all_tracks, all_points, view_keypoints


def _read_camera_dict(colmap_cameras, colmap_images):
    camera_dict = {}
    for image_id in colmap_images:
        image = colmap_images[image_id]

        img_name = image.name
        cam = colmap_cameras[image.camera_id]

        img_size = [cam.width, cam.height]
        params = list(cam.params)
        qvec = list(image.qvec)
        tvec = list(image.tvec)

        # w, h, fx, fy, cx, cy, s, qvec, tvec
        # camera_dict[img_name] = img_size + params + qvec + tvec

        fx, fy, cx, cy, s = params
        K = np.eye(4)
        K[0, 0] = fx
        K[0, 1] = s
        K[0, 2] = cx
        K[1, 1] = fy
        K[1, 2] = cy

        rot = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
        W2C = np.eye(4)
        W2C[:3, :3] = rot
        W2C[:3, 3] = np.array(tvec)
        
        camera_dict[img_name] = {
            'K': K.flatten().tolist(),
            'W2C': W2C.flatten().tolist(),
            'img_size': img_size
        }
        
    return camera_dict


def extract_camera_dict(sparse_dir, ext='.txt'):
    colmap_cameras, colmap_images, _ = read_model(sparse_dir, ext)
    camera_dict = _read_camera_dict(colmap_cameras, colmap_images)

    return camera_dict


def extract_all_to_dir(sparse_dir, out_dir, ext='.txt'):
    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, ext)
    camera_dict = _read_camera_dict(colmap_cameras, colmap_images)

    # output files
    os.makedirs(out_dir, exist_ok=True)
    camera_dict_file = os.path.join(out_dir, 'kai_cameras.json')
    xyz_file = os.path.join(out_dir, 'kai_points.txt')
    ply_file = os.path.join(out_dir, 'kai_points.ply')
    track_file = os.path.join(out_dir, 'kai_tracks.json')
    keypoints_file = os.path.join(out_dir, 'kai_keypoints.json')
    
    with open(camera_dict_file, 'w') as fp:
        json.dump(camera_dict, fp, indent=2, sort_keys=True)

    all_tracks, all_points, view_keypoints = _read_tracks(colmap_images, colmap_points3D)
    all_points = np.array(all_points)
    np.savetxt(xyz_file, all_points, header='# format: x, y, z, reproj_err, track_len, color(RGB)', fmt='%.6f')

    trimesh.PointCloud(vertices=all_points[:, :3].astype(np.float32),
                       colors=all_points[:, -3:].astype(np.uint8)).export(ply_file)

    with open(track_file, 'w') as fp:
        json.dump(all_tracks, fp)

    with open(keypoints_file, 'w') as fp:
        json.dump(view_keypoints, fp)

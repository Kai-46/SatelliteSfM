import multiprocessing
import cv2
import numpy as np
import json
import os
import imageio
import shutil


def warp_affine(img_src, affine_matrix):
    '''
    img_src: [H, W, 3] numpy array
    affine_matrix: [2, 3] numpy array
    '''
    height, width = img_src.shape[:2]

    # compute bounding box
    bbx = np.dot(affine_matrix, np.array([[0, width, width, 0],
                                          [0, 0, height, height],
                                          [1, 1, 1, 1]]))
    col_min = np.min(bbx[0, :])
    col_max = np.max(bbx[0, :])
    row_min = np.min(bbx[1, :])
    row_max = np.max(bbx[1, :])

    w = int(np.round(col_max - col_min + 1))
    h = int(np.round(row_max - row_min + 1))

    # add offset to the affine_matrix
    affine_matrix[0, 2] -= col_min
    affine_matrix[1, 2] -= row_min

    off_set = (-col_min, -row_min)

    # warp image
    img_dst = cv2.warpAffine(img_src, affine_matrix, (w, h))
    assert (h == img_dst.shape[0] and w == img_dst.shape[1])

    return img_dst, off_set, affine_matrix


def skew_correct_worker(in_dir, img_name, out_dir, keep_img_size=True):
    cam_dict = json.load(open(os.path.join(in_dir, 'cameras_adjusted', img_name[:-4]+'.json')))
    K = np.array(cam_dict['K']).reshape((4, 4))
    fx, s, cx = K[0, 0], K[0, 1], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]

    # compute homography and update s, cx
    norm_skew = s / fy
    cx = cx - s * cy / fy
    s = 0.

    # warp image
    affine_matrix = np.array([[1., -norm_skew, 0.],
                              [0., 1., 0.]])
    img_src = imageio.imread(os.path.join(in_dir, 'images' , img_name))
    orig_h, orig_w = img_src.shape[:2]
    img_dst, off_set, affine_matrix = warp_affine(img_src, affine_matrix)
    cx += off_set[0]
    cy += off_set[1]

    if keep_img_size:
        if img_dst.shape[0] > orig_h:
            img_dst = img_dst[:orig_h, :, :]
        elif img_dst.shape[0] < orig_h:
            img_dst = np.pad(img_dst, ((0, orig_h-img_dst.shape[0]), (0, 0), (0, 0)))

        if img_dst.shape[1] > orig_w:
            img_dst = img_dst[:, :orig_w, :]
        elif img_dst.shape[1] < orig_w:
            img_dst = np.pad(img_dst, ((0, 0), (0, orig_w-img_dst.shape[1]), (0, 0)))

    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'cameras'), exist_ok=True)
    imageio.imwrite(os.path.join(out_dir, 'images', img_name), img_dst)

    new_h, new_w = img_dst.shape[:2]
    cam_dict['img_size'] = [new_w, new_h]
    K[0, 1] = 0
    K[0, 2] = cx
    K[1, 2] = cy
    cam_dict['K'] = K.flatten().tolist()
    with open(os.path.join(out_dir, 'cameras', img_name[:-4]+'.json'), 'w') as fp:
        json.dump(cam_dict, fp, indent=2, sort_keys=True)


def skew_correct(in_dir, out_dir):
    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(pool_size)

    for img_name in os.listdir(os.path.join(in_dir, 'images')):
        if img_name.endswith('.png'):
            pool.apply_async(skew_correct_worker, (in_dir, img_name, out_dir))

    pool.close()
    pool.join()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Remove Skew from Images')
    parser.add_argument('--input_folder', type=str,
                    help='path to SatelliteSfM outputs')
    parser.add_argument('--output_folder', type=str,
                    help='path to output folder')

    args = parser.parse_args()
    skew_correct(args.input_folder, args.output_folder)


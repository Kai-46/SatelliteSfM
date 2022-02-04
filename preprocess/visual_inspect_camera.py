import os
import numpy as np
import shutil
import cv2
import multiprocessing
import imageio


def put_text(im, text, color, bottom_left=[10, 50], fontScale=None):
    orig_im_dtype = im.dtype
    if im.dtype == np.float32:
        im = np.clip(im*255., 0., 255.).astype(np.uint8)
        color = [int(x * 255.) for x in list(color)]

    if fontScale is None:
        fontScale = np.max(im.shape[:2]) / 1024.
        bottom_left = (int(fontScale * bottom_left[0]), int(fontScale * bottom_left[1]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    im = cv2.putText(im, text, bottom_left, font,
                     fontScale, color, thickness, cv2.LINE_AA)

    if orig_im_dtype == np.float32:
        im = im.astype(np.float32) / 255.
    return im


def compute_homography(ref_P, src_P, plane_vec):
    '''
        ref_P, src_P: 3x4 prjection matrix; numpy array
        plane_vec: (4,) vector; numpy array

        return:
            H: 3x3 homography to warp src image to ref image
    '''
    plane_vec = plane_vec.reshape((4, 1))
    plane_normal = plane_vec[:3, :]
    plane_constant = plane_vec[3, 0]

    ref_S = (ref_P[:3, :3] + np.matmul(ref_P[:3, 3:4], plane_normal.T) / plane_constant)
    src_S = (src_P[:3, :3] + np.matmul(src_P[:3, 3:4], plane_normal.T) / plane_constant)
    H = np.matmul(ref_S, np.linalg.inv(src_S))
    H = H / np.max(np.abs(H))   # increase numeric stability
    return H


def _warp_src_to_ref_singleplane(out_fpath, plane_vec, ref_im, ref_P, src_im, src_P, subarea=None):
    '''
        warp src image to the ref image pixel grid [ul_x, ul_y, w, h]
    '''
    ul_x, ul_y = 0, 0
    h, w = ref_im.shape[:2]
    if subarea is not None:
        ul_x, ul_y, w, h = subarea
        assert (0 <= ul_x <= ref_im.shape[1] - w and 0 <= ul_y <= ref_im.shape[0] - h)

    H = compute_homography(ref_P, src_P, plane_vec)
    translation = np.array([[1.0, 0.0, -ul_x], 
                            [0.0, 1.0, -ul_y], 
                            [0.0, 0.0, 1.0]])
    H = np.matmul(translation, H)

    warped_src_im = cv2.warpPerspective(src_im, H, (w, h), borderMode=cv2.BORDER_CONSTANT)
    imageio.imwrite(out_fpath, warped_src_im)


def warp_src_to_ref(out_dir, sweep_plane_sequence, ref_im, src_im, ref_P, src_P, subarea=None, max_processes=None):
    '''
        sweep_plane_sequence: [N, 4] numpy array; each row is plane normal+plane constant
        ref_im, src_im: uint8 image
        ref_P, src_P: 3x4 or 4x4 projection matrix
    '''
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    os.makedirs(os.path.join(out_dir, 'warped_src'))

    np.savetxt(os.path.join(out_dir, 'sweep_plane_sequence.txt'), sweep_plane_sequence)

    # write ref
    ul_x, ul_y = 0, 0
    h, w = ref_im.shape[:2]
    if subarea is not None:
        ul_x, ul_y, w, h = subarea
        assert (0 <= ul_x <= ref_im.shape[1] - w and 0 <= ul_y <= ref_im.shape[0] - h)
        ref_im = ref_im[ul_y:ul_y+h, ul_x:ul_x+w]
    imageio.imwrite(os.path.join(out_dir, 'ref_im.png'), ref_im)

    # warp src
    max_processes = multiprocessing.cpu_count() if max_processes is None else max_processes
    pool = multiprocessing.Pool(max_processes)
    results = []
    for i, plane_vec in enumerate(sweep_plane_sequence):
        out_fpath = os.path.join(out_dir, 'warped_src', '{}.png'.format(i))
        r = pool.apply_async(_warp_src_to_ref_singleplane, (out_fpath, plane_vec, ref_im, ref_P, src_im, src_P, subarea))
        results.append(r)
    [r.wait() for r in results]     # sync

    # create video
    frames = []
    for i, plane_vec in enumerate(sweep_plane_sequence):
        im = imageio.imread(os.path.join(out_dir, 'warped_src', '{}.png'.format(i)))
        im = ((im.astype(np.float32) + ref_im.astype(np.float32)) / 2.).astype(np.uint8)
        im = put_text(im, 'z={:.3f}m'.format(plane_vec[3]), color=(255, 0, 255))
        frames.append(im)
    imageio.mimwrite(os.path.join(out_dir, 'avg_ref_and_warpedsrc.mp4'), frames, fps=30)

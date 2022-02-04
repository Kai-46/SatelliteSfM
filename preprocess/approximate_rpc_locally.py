import numpy as np
from scipy import linalg
from icecream import ic

from preprocess.factorize_projection_matrix import factorize_projection_matrix
from preprocess.rpc_model import RPCModel
from preprocess.coordinate_system import latlonalt_to_enu


def _generate_samples(meta_dict, lat_minmax, lon_minmax, alt_minmax, 
                                 lat_N, lon_N, alt_N):
    '''
        meta_dict: see parse_tiff.py for information about meta_dict
        lat_minmax, lon_minmax, alt_minmax: (2,)
        lat_N, lon_N, alt_N: integers
    '''
    rpc_model = RPCModel(meta_dict)

    lon, lat, alt = np.meshgrid(np.linspace(lon_minmax[0], lon_minmax[1], lon_N),
                                np.linspace(lat_minmax[0], lat_minmax[1], lat_N),
                                np.linspace(alt_minmax[0], alt_minmax[1], alt_N))
    lon, lat, alt = lon.reshape((-1,)), lat.reshape((-1,)), alt.reshape((-1,))

    col, row = rpc_model.projection(lat, lon, alt)
    keep_mask = np.logical_and(col >= 0, row >= 0)
    keep_mask = np.logical_and(keep_mask, col < rpc_model.width)
    keep_mask = np.logical_and(keep_mask, row < rpc_model.height)
    return lat[keep_mask], lon[keep_mask], alt[keep_mask], col[keep_mask], row[keep_mask]


def _solve_projection_matrix(x, y, z, col, row, enable_debug=False):
    '''
        x, y, z, col, row: [N, ]; numpy array

        return:
            P: 3x4 projection matrix
    '''
    x, y, z, col, row = x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1)),\
                             col.reshape((-1, 1)), row.reshape((-1, 1))
    point_cnt = x.shape[0]
    all_ones = np.ones((point_cnt, 1))
    all_zeros = np.zeros((point_cnt, 4))

    A1 = np.hstack((x, y, z, all_ones,
                    all_zeros,
                    -col * x, -col * y, -col * z, -col * all_ones))
    A2 = np.hstack((all_zeros,
                    x, y, z, all_ones,
                    -row * x, -row * y, -row * z, -row * all_ones))
    A = np.vstack((A1, A2))
    u, s, vh = linalg.svd(A, full_matrices=False)
    P = np.real(vh[11, :]).reshape((3, 4))

    if enable_debug:
        tmp = np.matmul(np.hstack((x, y, z, all_ones)), P.T)
        approx_col = tmp[:, 0:1] / tmp[:, 2:3]
        approx_row = tmp[:, 1:2] / tmp[:, 2:3]
        pixel_err = np.sqrt((approx_row - row) ** 2 + (approx_col - col) ** 2)
        ic('# points: {}, approx. error (pixels): {}'.format(point_cnt, np.median(pixel_err)))
    return P


def approximate_rpc_locally(meta_dict, lat_minmax, lon_minmax, alt_minmax, 
                                       observer_lat, observer_lon, observer_alt,
                                       lat_N=100, lon_N=100, alt_N=50):
    '''
        meta_dict: see parse_tiff.py for information about meta_dict
        lat_minmax, lon_minmax, alt_minmax: (2,)
        observer_lat, observer_lon, observer_alt: float
        lat_N, lon_N, alt_N: integers
    '''
    lat, lon, alt, col, row = _generate_samples(meta_dict, lat_minmax, lon_minmax, alt_minmax, 
                                                           lat_N, lon_N, alt_N)

    assert (observer_alt < np.min(alt_minmax))

    e, n, u = latlonalt_to_enu(lat, lon, alt, observer_lat, observer_lon, observer_alt)

    P = _solve_projection_matrix(e, n, u, col, row)
    K, R, t = factorize_projection_matrix(P)

    K_4by4 = np.eye(4)
    K_4by4[:3, :3] = K
    W2C = np.eye(4)
    W2C[:3, :3] = R
    W2C[:3, 3] = t

    return K_4by4, W2C


if __name__  == '__main__':
    pass

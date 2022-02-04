import numpy as np
from scipy import linalg


def factorize_projection_matrix(P):
    '''
        factorize a 3x4 projection matrix P to K, R, t
        P: [3, 4]; numpy array
        
        return:
            K: [3, 3]; numpy array
            R: [3, 3]; numpy array
            t: [3, ]; numpy array
    '''
    K, R = linalg.rq(P[:, :3])
    t = linalg.lstsq(K, P[:, 3:4])[0]

    # fix the intrinsic and rotation matrix
    #   intrinsic matrix's diagonal entries must be all positive
    #   rotation matrix's determinant must be 1; otherwise there's an reflection component

    neg_sign_cnt = int(K[0, 0] < 0) + int(K[1, 1] < 0) + int(K[2, 2] < 0)
    if neg_sign_cnt == 1 or neg_sign_cnt == 3:
        K = -K

    new_neg_sign_cnt = int(K[0, 0] < 0) + int(K[1, 1] < 0) + int(K[2, 2] < 0)
    assert (new_neg_sign_cnt == 0 or new_neg_sign_cnt == 2)

    fix = np.diag((1, 1, 1))
    if K[0, 0] < 0 and K[1, 1] < 0:
        fix = np.diag((-1, -1, 1))
    elif K[0, 0] < 0 and K[2, 2] < 0:
        fix = np.diag((-1, 1, -1))
    elif K[1, 1] < 0 and K[2, 2] < 0:
        fix = np.diag((1, -1, -1))
    K = np.matmul(K, fix)
    R = np.matmul(fix, R)
    t = np.matmul(fix, t).reshape((-1,))

    assert (linalg.det(R) > 0)
    K /= K[2, 2]

    return K, R, t

import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.asarray(T)
    pts = np.asarray(points)

    single_point = pts.ndim == 1
    if single_point:
        pts = pts.reshape(1, -1)

    ones = np.ones((pts.shape[0], 1))
    pts_homogeneous = np.hstack([pts, ones])
    
    transformed_homogeneous = (T @ pts_homogeneous.T).T
    transformed_3d = transformed_homogeneous[:, :3]

    if single_point:
        return transformed_3d[0].tolist()
    return transformed_3d.tolist()
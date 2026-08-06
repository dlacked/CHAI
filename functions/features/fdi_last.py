import numpy as np

def get_centroid(polygon_points):
    """
    Calculate the centroid of a polygon.
    """
    pts = np.array(polygon_points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        pts = pts.reshape(-1, 2)
    return np.mean(pts, axis=0).tolist()

import numpy as np

def calculate_pca_rotation(centroids):
    """
    Calculate the PCA rotation angle and mean point (center) for a list of centroids via
    covariance eigen-decomposition. This is the single PCA used everywhere: the CSV pipeline
    (this function), the site's "PCA" overlay checkbox, and the ResNet Tooth model's meta
    features (see js/geometry.js calculatePCARotation, which mirrors this exactly).
    """
    pts = np.array(centroids)
    if len(pts) < 2:
        return np.mean(pts, axis=0) if len(pts) == 1 else np.zeros(2), 0.0

    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)

    if cov.ndim == 0 or np.allclose(cov, 0):
        return mean, 0.0

    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    angle = np.arctan2(evecs[1, 0], evecs[0, 0])

    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi
    return mean, angle

def get_theta_values(centroids, mean_pt=None, angle=None):
    """
    Given a list of centroids and a PCA (mean_pt, angle), projects each centroid into that
    frame and returns rotated coordinates + angle for each. If mean_pt/angle aren't supplied,
    computes them via calculate_pca_rotation() (the common case - pass them explicitly only if
    the caller already computed the same PCA elsewhere and wants to avoid redoing it).
    """
    if not centroids:
        return []

    if mean_pt is None or angle is None:
        mean_pt, angle = calculate_pca_rotation(centroids)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    
    results = []
    for idx, c in enumerate(centroids):
        rx = c[0] - mean_pt[0]
        ry = c[1] - mean_pt[1]
        
        tx = rx * cos_a - ry * sin_a
        ty = rx * sin_a + ry * cos_a
        
        angle_rad = np.arctan2(ty, tx)
        
        results.append({
            "idx": idx,
            "rotated_x": tx,
            "rotated_y": ty,
            "angle_rad": angle_rad,
            "angle_norm": angle_rad / np.pi
        })
        
    return results

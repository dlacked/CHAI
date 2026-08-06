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

def compute_arch_order(centroids, mean_pt, angle, is_upper):
    """
    Python port of js/geometry.js's computeArchOrder - builds the shortest possible path
    through all teeth via Held-Karp dynamic programming, starting from a fixed anatomical
    reference point (leftmost-side extreme, picked by jaw). Unlike a plain sort by rotated_x
    (which is what this CSV pipeline used before and breaks down wherever the arch bends back
    on itself), this finds the globally shortest Hamiltonian path, which reliably follows the
    curve's true left-to-right order. Keep in sync with computeArchOrder if that changes.

    centroids: list of (x, y) in original image coordinates.
    mean_pt, angle: the shared PCA (see calculate_pca_rotation).
    Returns a list of indices into `centroids`, in arch order.
    """
    n = len(centroids)
    if n == 0:
        return []

    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    def rotated_xy(pt):
        rx = pt[0] - mean_pt[0]
        ry = pt[1] - mean_pt[1]
        tx = rx * cos_a - ry * sin_a
        ty = rx * sin_a + ry * cos_a
        return tx, ty

    rot = [rotated_xy(c) for c in centroids]

    start_idx = 0
    target_val = -float('inf') if is_upper else float('inf')
    found_left = False
    for i, (rx, ry) in enumerate(rot):
        if rx < 0:  # Left side of the arch
            if is_upper:
                if ry > target_val:
                    target_val = ry
                    start_idx = i
                    found_left = True
            else:
                if ry < target_val:
                    target_val = ry
                    start_idx = i
                    found_left = True

    # Fallback if no vertex is on the left side
    if not found_left:
        min_rx = float('inf')
        for i, (rx, ry) in enumerate(rot):
            if rx < min_rx:
                min_rx = rx
                start_idx = i

    if n == 1:
        return [start_idx]

    # Precompute pairwise distances between every pair of teeth (original coordinates, same
    # as the JS version - rotation doesn't change Euclidean distance anyway).
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = centroids[i][0] - centroids[j][0]
            dy = centroids[i][1] - centroids[j][1]
            d = (dx * dx + dy * dy) ** 0.5
            dist[i][j] = d
            dist[j][i] = d

    # dp[mask][j] = shortest path that visits exactly the teeth in `mask` (always including
    # start_idx) and ends at tooth j. parent[mask][j] records the tooth visited right before j.
    size = 1 << n
    INF = float('inf')
    dp = [[INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]

    start_mask = 1 << start_idx
    dp[start_mask][start_idx] = 0.0

    for mask in range(size):
        if not (mask & start_mask):
            continue
        dp_mask = dp[mask]
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            cost_to_j = dp_mask[j]
            if cost_to_j == INF:
                continue
            dist_j = dist[j]
            for k in range(n):
                if mask & (1 << k):
                    continue
                next_mask = mask | (1 << k)
                new_cost = cost_to_j + dist_j[k]
                if new_cost < dp[next_mask][k]:
                    dp[next_mask][k] = new_cost
                    parent[next_mask][k] = j

    full_mask = size - 1
    best_end = start_idx
    best_cost = INF
    for j in range(n):
        if dp[full_mask][j] < best_cost:
            best_cost = dp[full_mask][j]
            best_end = j

    order = []
    mask = full_mask
    curr = best_end
    while curr != -1:
        order.append(curr)
        prev = parent[mask][curr]
        mask ^= (1 << curr)
        curr = prev
    order.reverse()
    return order

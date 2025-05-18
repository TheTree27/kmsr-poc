from collections import defaultdict
import numpy as np


def find_radii(points, centers):
    radii = []
    assigned_points = defaultdict(list)

    for point in points:
        distances = np.linalg.norm(centers - point, axis=1)
        nearest_center_idx = np.argmin(distances)
        assigned_points[nearest_center_idx].append(point)

    for idx, center in enumerate(centers):
        if assigned_points[idx]:
            assigned = np.array(assigned_points[idx])
            max_dist = np.max(np.linalg.norm(assigned - center, axis=1))
        else:
            max_dist = 0  # catch empty clusters with only the center
        radii.append(max_dist)

    return radii
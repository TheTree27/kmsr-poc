from collections import defaultdict
import numpy as np
from sklearn.metrics import DistanceMetric
from scipy.spatial import distance


# gonzalez.py finds centers, but doesn't calculate the radii
def find_radii(points, centers):
    radii = []
    assigned_points = defaultdict(list)

    # assign points to nearest center
    for point in points:
        distances = np.linalg.norm(centers - point, axis=1)
        nearest_center_idx = np.argmin(distances)
        assigned_points[nearest_center_idx].append(point)

    # calculate radii by finding the farthest point in each cluster
    for idx, center in enumerate(centers):
        if assigned_points[idx]:
            assigned = np.array(assigned_points[idx])
            max_dist = np.max(np.linalg.norm(assigned - center, axis=1))
        else:
            max_dist = 0  # catch empty clusters with only the center
        radii.append(max_dist)
    return radii

# because k-means and the heuristic provide centroids instead of centers from the given points
def find_centers(centroids, points):
    metric = DistanceMetric.get_metric('euclidean')
    dists = metric.pairwise(points, centroids)

    # assign each point to it's nearest centroid through a list that contains the centroids index for each point
    cluster_assignment = np.argmin(dists, axis=1)
    print(cluster_assignment)
    # assign each centroid to it's nearest point, making those the new centers
    center_idxs = np.argmin(dists, axis=0)
    print(center_idxs)
    exact_centers = [points[i] for i in center_idxs]
    # calculate new radii based on assignment with the new centers
    radii = np.zeros(len(exact_centers)) # for 0 radii as base case
    for i, point in enumerate(points):
        center_idx = cluster_assignment[i]
        center = exact_centers[center_idx]
        dist = distance.euclidean(point, center)
        if(dist > radii[center_idx]):
            radii[center_idx] = dist

    kmsr = sum(radii)

    return exact_centers, radii, kmsr





import numpy as np
from sklearn.metrics import DistanceMetric
from k_center import helper


def run(points, k):
    points = np.array(points)
    n = len(points)
    centers = []

    start_idx = np.random.choice(n)
    centers.append(points[start_idx])

    if (k == 1):
        radii = helper.find_radii(points, centers)
        return np.array(centers), radii, sum(radii)

    pairwise_dist = DistanceMetric.get_metric('euclidean')
    while (len(centers) < k):
        dists = np.min(pairwise_dist.pairwise(points, np.array(centers)),
                       axis=1)  # the axis attribute stops it from returning the same center over and over again

        farthest_idx = np.argmax(dists)
        centers.append(points[farthest_idx])

    radii = helper.find_radii(points, centers)

    return np.array(centers), radii, sum(radii)
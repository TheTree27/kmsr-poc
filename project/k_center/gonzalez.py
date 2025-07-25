# algorithm by T. F. Gonzalez, "Clustering to minimize the maximum intercluster distance," Theoretical Computer Science, vol. 38, pp. 293–306, Jan. 1985, doi: 10.1016/0304-3975(85)90224-5.


import numpy as np
from sklearn.metrics import DistanceMetric
from k_center import helper
import random


def run(points, k, seed=42):
    random.seed(seed)
    points = np.array(points)
    n = len(points)
    centers = []

    start_idx = np.random.choice(n)
    centers.append(points[start_idx])

    if (k == 1):
        radii = helper.find_radii(points, centers)
        return np.array(centers), radii, sum(radii)

    pairwise_dist = DistanceMetric.get_metric('euclidean') # only necessary metric for purely numeric data
    while (len(centers) < k):
        dists = np.min(pairwise_dist.pairwise(points, np.array(centers)),
                       axis=1)  # the axis attribute stops it from returning the same center over and over again, so it moves along that axis

        farthest_idx = np.argmax(dists)
        centers.append(points[farthest_idx])

    radii = helper.find_radii(points, centers)

    return np.array(centers), radii, sum(radii)
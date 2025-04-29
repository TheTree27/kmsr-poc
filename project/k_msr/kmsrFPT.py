from IPython import display
from kmsr import KMSR
from sklearn.datasets import make_blobs
from collections import defaultdict
import numpy as np
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances
import random
import itertools

dist = DistanceMetric.get_metric('euclidean')


def get_radius(point, centers, radii):
    idx = np.where(np.all(centers == point, axis=1))[0]  # this specifically only works for 2D data
    return radii[idx[0]] if len(idx) > 0 else 0


def dist_m(point_a, point_b, centers, radii):
    default_distance = euclidean(point_a, point_b)
    a_is_center = any(np.array_equal(point_a, center) for center in centers)
    b_is_center = any(np.array_equal(point_b, center) for center in centers)

    if a_is_center and b_is_center:
        return max(default_distance - get_radius(point_a, centers, radii) - get_radius(point_b, centers, radii), 0)
    elif a_is_center:
        return max(default_distance - get_radius(point_a, centers, radii), 0)
    elif b_is_center:
        return max(default_distance - get_radius(point_b, centers, radii), 0)
    else:
        return default_distance


def k_completion(points, centers, radii, k):
    if (k == len(centers)):
        return centers

    if (len(centers) == 0):
        centers.append(points[np.random.choice(len(points))])

    while (len(centers) < k):
        dists = np.min(pairwise_distances(points, metric=dist_m, centers=np.array(centers), radii=radii),
                       axis=1)  # the axis attribute stops it from returning the same center over and over again

        farthest_idx = np.argmax(dists)
        centers.append(points[farthest_idx])

    return centers

# estimation of k_msr
def kmsr_heuristic(points, k, random_state):
    kmsr = KMSR(n_clusters=k,
                algorithm="FPT-heuristic",
                epsilon=0.5,
                n_u=10000,
                n_test_radii=10,
                random_state=random_state)
    kmsr.fit(points)
    return kmsr.cluster_radii_

# this allows for tuples like [1, 1, 1, 1], that can lead to the same center being picked twice
def get_assignment_tuples(k):
    permutations = [list(p) for p in itertools.product(range(k), repeat=k)]
    return permutations


def guessing_radii(points, k):
    initial_radii = []
    possible_radii = []
    # first we need to guess the largest radius, to decrease our guessing interval from there
    # for that we need an estimated k_msr solution as upper bound

    kmsr = kmsr_heuristic(points, k, random.randint(0, k ^ k))

    # because we used a 1+epsilon approximation, we can simlify the equation a lot
    epsilon = 0.5
    for radius in kmsr:
        initial_radii.append((1 + epsilon) * radius)
    while (len(initial_radii) < k):
        initial_radii.append(
            0.0)  # we need exactly k radii for our purposes, this heuristic provides approximations with up to k radii so we add the rest as 0.0

    # we build multiple possible profiles by decreasing each radius step by step
    for i in range(len(initial_radii)):
        possible_radii.append([initial_radii[i]])
        if i < k - 1:
            while initial_radii[i] > initial_radii[i + 1] + epsilon:
                initial_radii[i] = initial_radii[i] - epsilon
                possible_radii[i].append(initial_radii[i])
        if i == k - 1:
            while initial_radii[i] > 0 + epsilon:
                initial_radii[i] = initial_radii[i] - epsilon
                possible_radii[i].append(initial_radii[i])

    # we add all permutations of these radii (the carthesian product)
    radii = [list(profile) for profile in itertools.product(*possible_radii)]
    return radii


def algorithm_2(points, k, radii, assignment_tuple):
    guessed_centers = k_completion(points=points, centers=[], radii=radii, k=k)
    for i in range(k):
        temp_centers = k_completion(points=points, centers=guessed_centers[:i], radii=radii, k=k)
        if(assignment_tuple[i]<i):
            radii[assignment_tuple[i]] += 3 * radii[i]
            radii[i] == 0
            temp_centers[i] == random.choice(points)
        elif(assignment_tuple[i] >= i):
            temp_centers[i] = temp_centers[assignment_tuple[i]]
            radii[i] = 3 * radii[i]
        guessed_centers = k_completion(points=points, centers=temp_centers[:i+1], radii=radii, k=k)
    return guessed_centers, radii




def approximate(points, k):
    print("Running approximation...")
    final_centers = []
    upper_bound = (2 + 0.5) * max([euclidean(a, b) for a in points for b in points]) # using a 2 + epsilon approximation the sum of all radii can't be larger than 2 + epsilon times the largest distance
    print("Guessing radii...")
    radius_profile_guesses = guessing_radii(points, k)
    assignment_tuples = get_assignment_tuples(k)
    final_radii = []
    print("Comparing guesses. This might take a while...")
    for profile in radius_profile_guesses:
        for a in assignment_tuples:
            guessed_centers, radius_profile = algorithm_2(points, k, profile, a)
            guessed_solution = (sum(radius_profile))
            if guessed_solution < upper_bound:
                final_centers = guessed_centers
                final_radii = radius_profile
                upper_bound = guessed_solution
    return final_centers, final_radii, upper_bound
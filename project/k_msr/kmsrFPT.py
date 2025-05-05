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
import math

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

# estimation of k_msr by
# N. Lenßen, "Experimentelle Analyse von Min-Sum-Radii Approximationsalgorithmen". Bachelorarbeit, Heinrich-Heine-Universität Düsseldorf, 2024.
# using
# L. Drexler, A. Hennes, A. Lahiri, M. Schmidt, and J. Wargalla, "Approximating Fair K-Min-Sum-Radii in Euclidean Space," in Lecture notes in computer science, 2023, pp. 119–133. doi: 10.1007/978-3-031-49815-2_9.
def kmsr_heuristic(points, k, random_state):
    kmsr = KMSR(n_clusters=k,
                algorithm="FPT-heuristic",
                epsilon=0.4, # needs to be between 0 and 0.5 for the approximation
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

    initial_radii = [] # will contain all guesses for the largest radius
    possible_radii = []



    # first we need to guess the largest radius, to decrease our guessing interval from there
    # for that we need an estimated k_msr solution as upper bound

    base_kmsr = kmsr_heuristic(points, k, random.randint(0, k ^ k)) # 1+epsilon approximation


    epsilon = 0.4 # to stay consistent with epsilon used in heuristic
    heuristic = [radius for radius in base_kmsr]
    max_heuristic = max(heuristic)
    beta = 1+epsilon
    r_1_upper = max_heuristic * k
    r_1_lower = max_heuristic / beta # divided by approximation factor of largest radius in heuristic
    initial_radii.append(r_1_lower)
    j = 1
    initial_radii.append(pow(1 + epsilon, j - 1) * (max_heuristic / beta))
    while (j < math.log(beta * k, 1+epsilon) and initial_radii[-1] < r_1_upper): # both conditions should mean the same
        initial_radii.append(pow(1+epsilon, j) * (max_heuristic/beta)) # if the actual radius is between this and the previous iteration, this one is at most 1+epsilon times worse
        j += 1
    possible_radii.append(initial_radii)


    i = 1
    for i in range(k):
        profile_section = []
        for radius in possible_radii[i]:
            current_interval = []
            j = i+1
            lower_bound = (epsilon / k) * radius
            current_interval.append(lower_bound)
            while (j < math.log(beta * k, 1+epsilon) and initial_radii[-1] < radius * k):
                current_interval.append(pow(1+epsilon, j) * (radius/beta))
                j += 1
            profile_section.append(current_interval)
        possible_radii.append(profile_section)



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
            guessed_solution = (sum(radius_profile)) # need to check if solution is viable
            if guessed_solution < upper_bound:
                final_centers = guessed_centers
                final_radii = radius_profile
                upper_bound = guessed_solution
    return final_centers, final_radii, upper_bound
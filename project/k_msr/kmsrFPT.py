from collections import defaultdict
import numpy as np
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances
import random
import itertools
import math
from k_center import gonzalez
from k_center import helper




pairwise_dist = DistanceMetric.get_metric('euclidean')


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

# this allows for tuples like [1, 1, 1, 1], that can lead to the same center being picked twice
def get_assignment_tuples(n, k):
    return list(itertools.combinations_with_replacement(range(n), k))



def guessing_radii(points, k):
    possible_radii_profiles = []

    # first we need to guess the largest radius, to decrease our guessing interval from there
    # for that we need an estimated k center solution as upper bound

    k_center, gonzalez_radii, gonzalez_solution = gonzalez.run(points, k)  # 2 approximation

    epsilon = 0.25  # use lower epsilon for more (and more precise), but slower results
    k_center_radii = helper.find_radii(points, k_center)
    max_radius = max(k_center_radii)
    beta = 2  # approximation factor
    j = 1

    # each profile is initialized with a sub interval of the largest possible solution, which would be one large radius aka the max k center radius times k
    possible_radii_profiles.append([pow(1 + epsilon, j - 1) * (max_radius / beta)])  # lower bound
    while (j < math.log(beta * k, 1 + epsilon) and possible_radii_profiles[
        -1] < max_radius * k):  # both conditions should mean the same
        possible_radii_profiles.append([pow(1 + epsilon, j) * (
                    max_radius / beta)])  # if the actual radius is between this and the previous iteration, this one is at most 1+epsilon times worse
        j += 1
    possible_radii_profiles.append([max_radius * k])  # upper bound

    # each profile is then appended with its own possible sub intervals (then the largest radius is put at the end)

    for profile in possible_radii_profiles:
        j = 2
        lower_bound = epsilon / k * profile[0]
        while (j <= math.log(k / epsilon, 1 + epsilon) and profile[-1] <= profile[0]):
            profile.append(pow(1 + epsilon, j) * lower_bound)
            j += 1
        # put largest radius of the interval at the end to have the profile sorted
        profile.append(profile[0])
        profile.pop(0)

    return possible_radii_profiles


def algorithm_2(points, k, radii, assignment_tuple):
    radii = [radii[i] for i in assignment_tuple] # because the guessed profile can (and almost always will) contain more than k radii
    assignment_tuple = [i for i in range(k)]

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


def all_points_covered(centers, radii, points):
    for point in points:
        covered = False
        for center, radius in zip(centers, radii): # assigns center at i to radius at i
            if euclidean(point, center) <= radius:
                covered = True
                break
            if not covered: return False
    return True




def approximate(points, k):
    print("Running approximation...")
    final_centers = []
    upper_bound = (2 + 0.5) * max([euclidean(a, b) for a in points for b in
                                   points])  # using a 2 + epsilon approximation the sum of all radii can't be larger than 2 + epsilon times the largest distance
    print("Guessing radii...")
    radius_profile_guesses = guessing_radii(points, k)
    assignment_tuples = get_assignment_tuples(len(radius_profile_guesses[0]), k)
    final_radii = []
    print("Comparing guesses. This might take a while...")
    i = 1
    for profile in radius_profile_guesses:
        print("Trying profile", i, "of", len(radius_profile_guesses))
        for a in assignment_tuples:
            guessed_centers, radius_profile = algorithm_2(points, k, profile.copy(),
                                                          a)  # .copy() ensures that profile isnt mutated in algorithm_2()
            guessed_solution = (sum(radius_profile))  # actual k-msr solution
            if (guessed_solution < upper_bound and all_points_covered(guessed_centers, radius_profile, points)):
                final_centers = guessed_centers
                final_radii = radius_profile
                upper_bound = guessed_solution
        i += 1
    return final_centers, final_radii, upper_bound
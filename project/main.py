# script to manually call algorithms with desired k and data set

import argparse
import time

from k_msr import kmsrILP, kmsrHeuristic
from k_msr import kmsrFPT
import k_center.gonzalez
import k_center.k_means_ppCenter
import k_center.kmsrHeuristicCenter
from _io___.readData import read
from k_means import k_means_pp


def ilp(points, k):
    print("Running ILP")
    return kmsrILP.run_model(points, k)

def fpt(points, k):
    print("Running FPT")
    return kmsrFPT.approximate(points, k)

# T. F. Gonzalez, "Clustering to minimize the maximum intercluster distance," Theoretical Computer Science, vol. 38, pp. 293–306, Jan. 1985, doi: 10.1016/0304-3975(85)90224-5.
def gonzalez(points, k, seed=None):
    print("Running Gonzalez")
    return k_center.gonzalez.run(points, k)

# N. Lenßen, "Experimentelle Analyse von Min-Sum-Radii Approximationsalgorithmen". Bachelorarbeit, Heinrich-Heine-Universität Düsseldorf, 2024.
def heuristic(points, k):
    print("Running FPT heuristic")
    return kmsrHeuristic.run(points, k)

# D. Arthur and S. Vassilvitskii, "k-means++: the advantages of careful seeding," Symposium on Discrete Algorithms, pp. 1027–1035, Jan. 2007, doi: 10.5555/1283383.1283494.
def k_means(points, k, seed=42):
    print("Running K-Means")
    return k_means_pp.run(points, k)

def k_means_center(points, k, seed=42):
    print("Running K-MeansCenter")
    return k_center.k_means_ppCenter.run(points, k, seed)

def heuristic_center(points, k, seed=42):
    print("Running HeuristicCenter")
    return k_center.k_means_ppCenter.run(points, k, seed)


algorithms = {
    'ILP': ilp,
    'FPT': fpt,
    'GONZALEZ': gonzalez,
    'HEURISTIC': heuristic,
    'K-MEANS': k_means,
    'K-MEANS-CENTER': k_means_center,
    'HEURISTIC-CENTER': heuristic_center,
}

def parse_input():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Select which k_msr algorithm to use',
        epilog='')  # at the bottom of help
    parser.add_argument('filename')
    parser.add_argument('k', type=int, help='Number of centers')
    parser.add_argument('algorithm', help='Which algorithm to use, eg. "ILP"')
    return parser

# to be called by other scripts, supports choice of random seed (in this case pipeline.py)
def _main(algorithm, k, filename, seed=None):
    algorithm = algorithms[algorithm]
    points = read("data_sets/" + filename)
    if seed:
        return algorithm(points, k, seed)
    return algorithm(points, k)

# calling the algorithm with k value on data set specified in cl arguments
def main():
    start_time = time.time() # track time of execution
    parser = parse_input()
    args = parser.parse_args()
    algorithm = algorithms[args.algorithm]
    k = args.k
    points = read(args.filename)
    if algorithm:
        centers, radii, sum_of_radii = algorithm(points, k)
        print("Centers:", centers)
        print("Radii:", radii)
        print("kMSR:", sum_of_radii)
    else:
        print("No valid algorithm selected")
    print("Time elapsed: ", time.time() - start_time, "seconds")

if __name__ == '__main__':
    main()

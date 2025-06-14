import argparse
import time

from k_msr import kmsrILP, kmsrHeuristic
from k_msr import kmsrFPT
import k_center.gonzalez
from _io___.readData import read
from k_means import k_means_pp


def ilp(points, k):
    print("Running ILP")
    return kmsrILP.run_model(points, k)

def fpt(points, k):
    print("Running FPT")
    return kmsrFPT.approximate(points, k)

# T. F. Gonzalez, "Clustering to minimize the maximum intercluster distance," Theoretical Computer Science, vol. 38, pp. 293–306, Jan. 1985, doi: 10.1016/0304-3975(85)90224-5.
def gonzalez(points, k):
    print("Running Gonzalez")
    return k_center.gonzalez.run(points, k)

# N. Lenßen, "Experimentelle Analyse von Min-Sum-Radii Approximationsalgorithmen". Bachelorarbeit, Heinrich-Heine-Universität Düsseldorf, 2024.
def heuristic(points, k):
    print("Running FPT heuristic")
    return kmsrHeuristic.run(points, k)

def k_means(points, k):
    print("Running K-Means")
    return k_means_pp.run(points, k)


algorithms = {
    'ILP': ilp,
    'FPT': fpt,
    'GONZALEZ': gonzalez,
    'HEURISTIC': heuristic,
    'K-MEANS': k_means,
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

def _main(algorithm, k, filename):
    algorithm = algorithms[algorithm]
    points = read("data_sets/" + filename)
    return algorithm(points, k)

def main():
    start_time = time.time()
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

import argparse

from k_msr import kmsrILP
from k_msr import kmsrFPT
import k_center.gonzalez
from readData import read


def ilp(points, k):
    print("Running ILP")
    return kmsrILP.run_model(points, k)

def fpt(points, k):
    print("Running FPT")
    return kmsrFPT.approximate(points, k)

def gonzalez(points, k):
    print("Running Gonzalez")
    return k_center.gonzalez.run(points, k)

algorithms = {
    'ILP': ilp,
    'FPT': fpt,
    'GONZALEZ': gonzalez
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



def main():
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

if __name__ == '__main__':
    main()

import argparse

from kmsr import kmsrILP
from readData import read


def ilp(points, k):
    print("Running ILP")
    return kmsrILP.run_model(points, k)

algorithms = {
    'ILP': ilp
}

def parse_input():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Select which kmsr algorithm to use',
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
        centers, radii = algorithm(points, k)
        print(centers)
        print(radii)
    else:
        print("No valid algorithm selected")

if __name__ == '__main__':
    main()

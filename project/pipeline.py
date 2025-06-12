from datetime import datetime
from main import _main
from io.writer import *

algorithms = ["ILP", "FPT", "GONZALEZ", "HEURISTIC", "K-MEANS"]
ks = [1, 2, 3, 4, 10, 100]
data_sets = ["iris.data", "rl5934.txt", "ruspini.csv"]


def main():
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    write_header(filename)
    for algorithm in algorithms:
        for data_set in data_sets:
            for k in ks:
                print("Computing", algorithm, "on", data_set, "with k=", k)
                centers, radii, sum_of_radii = _main(algorithm, data_set, k)
                write_result(algorithm, centers, radii, sum_of_radii)
    print("Done")


if __name__ == '__main__':
    main()
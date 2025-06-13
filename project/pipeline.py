from datetime import datetime
from main import _main
from _io_ import writer
from termcolor import colored

algorithms = ["ILP", "FPT", "GONZALEZ", "HEURISTIC", "K-MEANS"]
ks = [1, 2, 3, 4, 10, 100]
data_sets = ["iris.data", "rl5934.txt", "ruspini.csv"]


def main():
    for algorithm in algorithms:
        filename = "results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-" + algorithm + ".csv"
        writer.write_header(filename)
        for data_set in data_sets:
            if data_set == "rl5934.txt" and algorithm == "ILP":
                continue
            for k in ks:
                print(colored("Computing " + algorithm + " on " + data_set + " with k = " + str(k), 'green'))
                centers, radii, sum_of_radii = _main(algorithm, k, data_set)
                writer.write_result(filename, k, centers, radii, sum_of_radii, data_set)
    print(colored("Done", 'green'))


if __name__ == '__main__':
    main()
# script used to get kmsr values for the thesis
from datetime import datetime
from main import _main
from _io___ import writer
from termcolor import colored

# intended values in comments
algorithms = ["K-MEANS-CENTER", "HEURISTIC-CENTER"] # ["ILP", "FPT", "GONZALEZ", "HEURISTIC", "K-MEANS", "K-MEANS-CENTER", "HEURISTIC-CENTER"
ks = [1, 2, 3, 4, 5, 10, 100] # [1, 2, 3, 4, 5, 10, 100]
data_sets = ["Car.csv", "ali535.txt", "ch150.txt", "iris.data", "rl5934.txt", "ruspini.csv"] # ["Car.csv", "ali535.txt", "ch150.txt", "iris.data", "rl5934.txt", "ruspini.csv"]


def main():
    for algorithm in algorithms:
        # create sortable files for results of each algorithm
        filename = "results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-" + algorithm + ".csv"
        writer.write_header(filename)
        # filter out combinations that would take too long, especially on the larger data sets
        for data_set in data_sets:
            if data_set == "rl5934.txt" and algorithm == "ILP":
                continue
            for k in ks:
                if (algorithm == "HEURISTIC" or algorithm == "HEURISTIC-CENTER") and k>10:
                    continue
                if (algorithm == "HEURISTIC" or algorithm == "HEURISTIC-CENTER") and data_set == "rl5934.txt" and k>=10:
                    continue
                if algorithm == "FPT" and k >= 5:
                    continue
                if algorithm == "FPT" and data_set in ["ali535.txt", "rl5934.txt"] and k >= 4:
                    continue
                # for gonzalez and kmeans++, try out 100 different seeds to find variance
                if algorithm == "GONZALEZ" or algorithm == "K-MEANS":
                    print(colored("Computing " + algorithm + " on " + data_set + " with k = " + str(k), 'green'))
                    for i in range(1, 101):
                        centers, radii, sum_of_radii = _main(algorithm, k, data_set, seed=i)
                        writer.write_result(filename, k, centers, radii, sum_of_radii, data_set) # output for centers is partially difficult because of lists of np.arrays, but can be easily cleaned up during evaluation
                    continue
                print(colored("Computing " + algorithm + " on " + data_set + " with k = " + str(k), 'green'))
                centers, radii, sum_of_radii = _main(algorithm, k, data_set) # call function in main.py for each algorithm/k/data set
                writer.write_result(filename, k, centers, radii, sum_of_radii, data_set)
    print(colored("Done", 'green'))


if __name__ == '__main__':
    main()
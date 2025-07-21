# meant as a really simple way to get some data about ILP realized run time
import time
from sklearn.datasets import make_blobs
from k_msr import kmsrILP
import csv

list_of_k = [1,2,3,4,5,10,100]
list_of_n = [10,100,500]


def main():
    filename = "results/ILPTime.csv"
    with open(filename, mode='w', newline='') as file:
        csv.writer(file).writerow(["n", "k", "seconds"])

    for n in list_of_n:
        for k in list_of_k:
            if (n==500 and k>1):
                break
            points, cluster_membership = make_blobs(
                n_samples=n,
                n_features=2,
                centers=k,
                random_state=42
            )
            start_time = time.time()
            centers, radii, sum_of_radii = kmsrILP.run_model(points, k)
            time_elapsed = time.time() - start_time
            with open(filename, mode='a', newline='') as file:
                csv.writer(file).writerow([n, k, time_elapsed])

if __name__ == '__main__':
    main()
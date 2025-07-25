# algorithm by D. Arthur and S. Vassilvitskii, "k-means++: the advantages of careful seeding," Symposium on Discrete Algorithms, pp. 1027–1035, Jan. 2007, doi: 10.5555/1283383.1283494.
# implemented by N. Lenßen, "Experimentelle Analyse von Min-Sum-Radii Approximationsalgorithmen". Bachelorarbeit, Heinrich-Heine-Universität Düsseldorf, 2024.
from kmsr import KMSR
from k_center import helper

# prevent unwanted output
import sys
import io

def run(points, k, seed=42):
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    # provides centroids in space, not centers from the given points
    kmpp = KMSR(n_clusters=k,
                     algorithm="KMeans",
                     epsilon=0.5,
                     n_u=10000,
                     n_test_radii=10,
                     random_state=seed)
    kmpp.fit(points)
    sys.stdout = save_stdout
    # select nearest points to centroids as centers, calculate new radii accordingly
    centers, radii, kmsr = helper.find_centers(kmpp.cluster_centers_, points)
    return centers, radii, kmsr
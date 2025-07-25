# based on the paper L. Drexler, A. Hennes, A. Lahiri, M. Schmidt, and J. Wargalla, "Approximating Fair K-Min-Sum-Radii in Euclidean Space," in Lecture notes in computer science, 2023, pp. 119–133. doi: 10.1007/978-3-031-49815-2_9.
# implementation by N. Lenßen, "Experimentelle Analyse von Min-Sum-Radii Approximationsalgorithmen". Bachelorarbeit, Heinrich-Heine-Universität Düsseldorf, 2024.
from kmsr import KMSR
from k_center import helper

# prevent unwanted output
import sys
import io


# provides centroids in space, not centers from the given points
def run(points, k):
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    heuristic = KMSR(n_clusters=k,
                     algorithm="FPT-heuristic",
                     epsilon=0.5,
                     n_u=10000,
                     n_test_radii=10,
                     random_state=42)
    heuristic.fit(points)
    sys.stdout = save_stdout
    # select nearest points to centroids as centers, calculate new radii accordingly
    centers, radii, kmsr = helper.find_centers(heuristic.cluster_centers_, points)
    return centers, radii, kmsr
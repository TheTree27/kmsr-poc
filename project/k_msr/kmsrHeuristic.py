from kmsr import KMSR

# prevent unwanted output
import sys
import io



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
    return heuristic.cluster_centers_, heuristic.cluster_radii_, sum(heuristic.cluster_radii_)
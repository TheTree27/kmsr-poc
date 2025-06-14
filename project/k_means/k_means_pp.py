from kmsr import KMSR

# prevent unwanted output
import sys
import io


# provides centroids in space, not centers from the given points
def run(points, k):
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    kmpp = KMSR(n_clusters=k,
                     algorithm="KMeans",
                     epsilon=0.5,
                     n_u=10000,
                     n_test_radii=10,
                     random_state=42)
    kmpp.fit(points)
    sys.stdout = save_stdout
    return kmpp.cluster_centers_, kmpp.cluster_radii_, sum(kmpp.cluster_radii_)
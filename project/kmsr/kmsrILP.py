import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics import DistanceMetric
from scipy.spatial import distance as dist

def run_model(points, k):

    distances = DistanceMetric.get_metric('euclidean')
    radii = distances.pairwise(points)

    m = gp.Model("kmsr")
    y = m.addVars(len(points), len(points), vtype=GRB.BINARY, name="Y")

    for n in range(len(points)):
        m.addConstr(gp.quicksum([y[i, j] for j in range(len(radii)) for i in range(len(radii)) if
                                 radii[i][j] >= dist.euclidean(points[n], points[i])]) >= 1, "every_point_covered")
    m.addConstr(gp.quicksum([y[i, j] for i in range(len(radii)) for j in range(len(radii[i]))]) == k,
                "select_k_centers")

    m.optimize()
    final_centers = [points[i] for i in range(len(points)) for j in range(len(radii[i])) if y[i, j].x == 1]
    final_radii = [radii[i][j] for i in range(len(radii)) for j in range(len(radii[i])) if y[i, j].x == 1]

    return final_centers, final_radii
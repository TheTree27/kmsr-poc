# integer linear program to solve k-min-sum-radii
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics import DistanceMetric
from scipy.spatial import distance as dist

def run_model(points, k):

    print("Calculating distances")
    distances = DistanceMetric.get_metric('euclidean')
    print("Distances calculated")
    radii = distances.pairwise(points) # all possible radii. value at i,j means distance between points at i and j

    print("Initializing model...")
    m = gp.Model("k_msr")
    y = m.addVars(len(points), len(points), vtype=GRB.BINARY, name="Y")

    # objective:
    m.setObjective(gp.quicksum([y[i, j] * radii[i][j] for i in range(len(radii)) for j in range(len(radii[i]))]),
                   GRB.MINIMIZE)  # minimization of the sum of active y_ij multiplied with their respective radii

    print("Adding constraints...")
    # each point needs to be covered, so for each point there needs to be one y_ij set to 1, whose radius is at least as large
    # as the points distance to the center at i
    for n in range(len(points)):
        m.addConstr(gp.quicksum([y[i, j] for j in range(len(radii)) for i in range(len(radii)) if
                                 radii[i][j] >= dist.euclidean(points[n], points[i])]) >= 1, "every_point_covered")
    # there need to be k centers
    m.addConstr(gp.quicksum([y[i, j] for i in range(len(radii)) for j in range(len(radii[i]))]) == k,
                "select_k_centers")

    print("Solving model. This could take some time... ")
    # gurobi brute forces this for smaller data sets, but uses heuristics to eliminate impossible solutions before computation for larger data sets (e.g. ali535)
    m.optimize()
    print("Model solved!")
    final_centers = [points[i] for i in range(len(points)) for j in range(len(radii[i])) if y[i, j].x == 1]
    final_radii = [radii[i][j] for i in range(len(radii)) for j in range(len(radii[i])) if y[i, j].x == 1]

    return final_centers, final_radii, sum(final_radii)
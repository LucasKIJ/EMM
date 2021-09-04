import numpy as np
from numpy import linalg
from scipy import sparse
import qdldl

from emm.losses import *
from emm.regularizers import *

# Gurobi optimisation
import gurobipy as gp
from gurobipy import GRB

# CVX optimisation
import cvxpy as cp


def cvx(F, losses, reg, lam=1):

    m, n = F.shape
    w = cp.Variable(n)
    # Initialise objective
    objective = 0
    # Initialise constraints
    constrs = [cp.sum(w) == 1, w >= 0]
    ct_cum = 0
    for l in losses:
        f = F[ct_cum : (ct_cum + l.m), :]
        ct_cum += l.m
        expr, type_expr = l.cvx(f, w)
        if type_expr == "o":
            objective += expr
        if type_expr == "c":
            constrs += expr
    expr, type_expr = reg.cvx(w, lam)
    if type_expr == "o":
        objective += expr
    if type_expr == "c":
        constrs += expr

    cp.Problem(cp.Minimize(objective), constrs).solve()

    return w.value


# ## Gurobi ##
# def gurobi(F, losses, reg, lam):
#     m, n = F.shape
#     # Desired marginal values
#     fdes = np.array([l.fdes for l in losses]).flatten()

#     # Create model
#     md = gp.Model("emm1")

#     # Suppress console output
#     md.Params.LogToConsole = 0

#     # Create variables
#     w = md.addVars(n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
#     wlogw = md.addVars(n, lb=-gp.GRB.INFINITY, vtype=GRB.CONTINUOUS, name="wlogw")
#     # Marginals
#     f = md.addVars(m, vtype=GRB.CONTINUOUS, name="f")

#     # Create constraints
#     # sum(w) == 1
#     md.addConstr(gp.quicksum(w[i] for i in range(n)) == 1, "c0")
#     # w >= 0 for all w
#     md.addConstrs((w[i] >= 0 for i in range(n)), "c1")
#     # f = Fw
#     md.addConstrs(
#         (f[i] == gp.quicksum(F[i, j] * w[j] for j in range(n)) for i in range(m))
#     )

#     # Set up entropy regulariser
#     if reg.__class__ == EntropyRegularizer().__class__:
#         # Create x-points and y-points for approximation of y = x*log(x)
#         xs = [0.01 * i for i in range(101)]
#         ys = [p * np.log(p) if p != 0 else 0 for p in xs]

#         # Regulariser
#         for i in range(n):
#             md.addGenConstrPWL(w[i], wlogw[i], xs, ys, name="wlogw")
#         regu = -gp.quicksum(wlogw[i] for i in range(n))

#     # Zero regulariser
#     if reg.__class__ == ZeroRegularizer().__class__:
#         regu = 0

#     # Squared loss objective
#     loss = gp.quicksum((f[i] - fdes[i]) * (f[i] - fdes[i]) for i in range(m))
#     md.setObjective(loss + lam * regu, GRB.MINIMIZE)

#     # # Optimize model
#     md.optimize()
#     # for v in md.getVars():
#     #     print('%s %g' % (v.varName, v.x))
#     print("Obj: %g" % md.objVal)

#     return np.array([w[i].x for i in range(n)])


if __name__ == "__main__":
    np.random.seed(1)
    from losses import *
    from regularizers import *

    n = 100
    m = 20
    F = np.random.randn(m, n)
    fdes1 = np.random.randn(m // 2)
    fdes2 = np.random.randn(m // 2)
    losses = [
        LeastSquaresLoss(fdes1),
        InequalityLoss(fdes2, -1 * np.ones(m // 2), 1 * np.ones(m // 2)),
    ]
    reg = EntropyRegularizer()

    import cvxpy as cp

    w = cvx(F, losses, reg)
    np.testing.assert_allclose(w.value, sol["w"], atol=1e-3)

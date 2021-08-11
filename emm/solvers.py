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

## Admm optimisation
def _projection_simplex(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def admm(
    F,
    losses,
    reg,
    lam,
    rho=50,
    maxiter=5000,
    eps=1e-6,
    warm_start={},
    verbose=False,
    eps_abs=1e-5,
    eps_rel=1e-5,
):
    m, n = F.shape
    ms = [l.m for l in losses]

    if "f" in warm_start.keys():
        f = warm_start["f"]
    else:
        f = np.array(F.mean(axis=1)).flatten()

    if "w" in warm_start.keys():
        w = warm_start["w"]
    else:
        w = np.ones(n) / n

    if "w_bar" in warm_start.keys():
        w_bar = warm_start["w_bar"]
    else:
        w_bar = np.ones(n) / n

    if "w_tilde" in warm_start.keys():
        w_tilde = warm_start["w_tilde"]
    else:
        w_tilde = np.ones(n) / n

    if "y" in warm_start.keys():
        y = warm_start["y"]
    else:
        y = np.zeros(m)

    if "z" in warm_start.keys():
        z = warm_start["z"]
    else:
        z = np.zeros(n)

    if "u" in warm_start.keys():
        u = warm_start["u"]
    else:
        u = np.zeros(n)

    Q = sparse.bmat([[2 * sparse.eye(n), F.T], [F, -sparse.eye(m)]])
    factor = qdldl.Solver(Q)

    if verbose:
        print(u"Iteration     | ||r||/\u03B5_pri | ||s||/\u03B5_dual")

    w_best = None
    best_objective_value = float("inf")

    for k in range(maxiter):
        ct_cum = 0
        for l in losses:
            f[ct_cum : ct_cum + l.m] = l.prox(
                F[ct_cum : ct_cum + l.m] @ w - y[ct_cum : ct_cum + l.m], 1 / rho
            )
            ct_cum += l.m

        w_tilde = reg.prox(w - z, lam / rho)
        w_bar = _projection_simplex(w - u)

        rhs = np.append(F.T @ (f + y) + w_tilde + z + w_bar + u, np.zeros(m))
        w_new = factor.solve(rhs)[:n]
        s = rho * np.concatenate([F @ w_new - f, w_new - w, w_new - w])
        w = w_new

        y = y + f - F @ w
        z = z + w_tilde - w
        u = u + w_bar - w

        r = np.concatenate([f - F @ w, w_tilde - w, w_bar - w])

        p = m + 2 * n
        Ax_k_norm = np.linalg.norm(np.concatenate([f, w_tilde, w_bar]))
        Bz_k_norm = np.linalg.norm(np.concatenate([w, w, w]))
        # y = rho * u
        ATy_k_norm = np.linalg.norm(rho * np.concatenate([y, z, u]))
        eps_pri = np.sqrt(p) * eps_abs + eps_rel * max(Ax_k_norm, Bz_k_norm)
        eps_dual = np.sqrt(p) * eps_abs + eps_rel * ATy_k_norm

        s_norm = np.linalg.norm(s)
        r_norm = np.linalg.norm(r)
        if verbose and k % 50 == 0:
            print(
                "It %03d / %03d | %8.5e | %8.5e"
                % (k, maxiter, r_norm / eps_pri, s_norm / eps_dual)
            )

        if isinstance(reg, BooleanRegularizer):
            ct_cum = 0
            objective = 0.0
            for l in losses:
                objective += l.evaluate(F[ct_cum : ct_cum + l.m] @ w_tilde)
                ct_cum += l.m
            if objective < best_objective_value:
                if verbose:
                    print(
                        "Found better objective value: %3.5f -> %3.5f"
                        % (best_objective_value, objective)
                    )
                best_objective_value = objective
                w_best = w_tilde

        if r_norm <= eps_pri and s_norm <= eps_dual:
            break

    if not isinstance(reg, BooleanRegularizer):
        w_best = w_bar

    return {
        "f": f,
        "w": w,
        "w_bar": w_bar,
        "w_tilde": w_tilde,
        "y": y,
        "z": z,
        "u": u,
        "w_best": w_best,
    }


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

    sol = admm(F, losses, reg, 1, verbose=True)

    import cvxpy as cp

    w = cvx(F, losses, reg)
    np.testing.assert_allclose(w.value, sol["w"], atol=1e-3)

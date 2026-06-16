import numpy as np
import logging
from scipy.optimize import linprog

# Basic console logging for script-style execution.
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def solve_target_optimization(weights, lower_bounds, upper_bounds, target):
    """
    Minimize |SUM(w_i * x_i) - T|
    Subject to: L_i <= x_i <= min(U_i, 1.0)

    Parameters
    ----------
    weights       : list of w_i  (entity weights, must sum to ~1)
    lower_bounds  : list of L_i  (lower boundary per entity)
    upper_bounds  : list of U_i  (upper boundary per entity)
    target        : float T      (overall weighted target)

    Returns
    -------
    dict with solution details
    """
    n = len(weights)
    w = np.array(weights)
    L = np.array(lower_bounds)
    U = np.array(upper_bounds)

    # ── Decision variables: [x_0, ..., x_{n-1}, t] ──────────────────
    # Total variables = n entities + 1 slack (t)

    # Objective: minimize t  →  coefficients = [0, 0, ..., 0, 1]
    c = np.zeros(n + 1)
    c[-1] = 1.0  # coefficient of t

    # ── Inequality constraints  A_ub @ vars <= b_ub ──────────────────
    #
    # Constraint 1:  SUM(w_i * x_i) - t <= T
    #   → [w_0, w_1, ..., w_{n-1}, -1] @ [x, t] <= T
    #
    # Constraint 2: -SUM(w_i * x_i) - t <= -T
    #   → [-w_0, -w_1, ..., -w_{n-1}, -1] @ [x, t] <= -T

    row1 = np.append(w, -1.0)    # SUM(w*x) - t <= T
    row2 = np.append(-w, -1.0)   # -SUM(w*x) - t <= -T

    A_ub = np.vstack([row1, row2])
    b_ub = np.array([target, -target])

    # ── Bounds ───────────────────────────────────────────────────────
    # x_i: [L_i, min(U_i, 1.0)]
    # t:   [0, None]  (t >= 0, unbounded above)

    bounds = [(L[i], min(U[i], 1.0)) for i in range(n)] + [(0, None)]

    # ── Solve ─────────────────────────────────────────────────────────
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs"  # HiGHS — fastest and most robust scipy LP solver
    )

    if not result.success:
        raise ValueError(f"Solver failed: {result.message}")

    x_opt = result.x[:n]   # optimal proposals
    t_opt = result.x[-1]   # residual (how close we got to target)
    achieved = float(w @ x_opt)

    return {
        "proposals":       x_opt.tolist(),
        "achieved_target": achieved,
        "residual":        t_opt,
        "exact_match":     abs(achieved - target) < 1e-6,
        "status":          result.message,
    }


# ── Example: 9 Lufthansa Hub Areas ────────────────────────────────────────────

areas = ["DE Hub", "BE Hub", "CH Hub", "AT Hub", "FRA", "VIE", "MUC", "ZRH", "BRU"]

weights = [0.30, 0.08, 0.07, 0.05, 0.18, 0.12, 0.10, 0.06, 0.04]  # must sum to 1.0
wop     = [0.78, 0.81, 0.83, 0.80, 0.76, 0.82, 0.79, 0.84, 0.85]  # current WOP baseline
lower   = [0.75, 0.78, 0.80, 0.77, 0.73, 0.79, 0.76, 0.81, 0.82]  # L_i
upper   = [0.92, 0.95, 0.96, 0.93, 0.90, 0.94, 0.91, 0.96, 0.97]  # U_i
target  = 0.82                                                        # T

result = solve_target_optimization(weights, lower, upper, target)

# ── Print Results ──────────────────────────────────────────────────────────────
logger.info(f"{'Area':<12} {'Weight':>8} {'WOP':>6} {'Lower':>7} {'Upper':>7} {'Proposal':>10}")
logger.info("-" * 55)
for i, area in enumerate(areas):
    logger.info(f"{area:<12} {weights[i]:>8.3f} {wop[i]:>6.3f} "
          f"{lower[i]:>7.3f} {upper[i]:>7.3f} "
          f"{result['proposals'][i]:>10.4f}")

logger.info("-" * 55)
logger.info(f"{'Target':<30} {target:.4f}")
logger.info(f"{'Achieved weighted sum':<30} {result['achieved_target']:.4f}")
logger.info(f"{'Residual |achieved - T|':<30} {result['residual']:.6f}")
logger.info(f"{'Exact match':<30} {result['exact_match']}")
logger.info(f"{'Solver status':<30} {result['status']}")
""" Sample with values for stations. """

import numpy as np
import logging
from scipy.optimize import minimize

# Basic console logging for script-style execution.
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Example data
areas = ["A", "B", "C", "D", "E"]
wop = np.array([0.82, 0.65, 0.55, 0.40, 0.30])
weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
lower = np.array([0.50, 0.50, 0.50, 0.50, 0.50])
upper = np.array([0.85, 0.85, 0.85, 0.85, 0.85])
target = 0.72

# objective: stay as close as possible to original WOP
def objective(x):
    return np.sum((x - wop) ** 2)

constraints = [
    {
        "type": "eq",
        "fun": lambda x: np.dot(weights, x) - target
    }
]

bounds = list(zip(lower, upper))

result = minimize(
    objective,
    x0=wop,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)


# ── Print Results ──────────────────────────────────────────────────────────────
logger.info(f"{'Area':<12} {'Weight':>8} {'WOP':>6} {'Lower':>7} {'Upper':>7} {'JAC':>10} {'Proposal':>10}")
logger.info("-" * 66)
for i, area in enumerate(areas):
    logger.info(f"{area:<12} {weights[i]:>8.3f} {wop[i]:>6.3f} "
                f"{lower[i]:>7.3f} {upper[i]:>7.3f} "
                f"{result['jac'][i]:>10.4f} "
                f"{result['x'][i]:>10.4f}")

logger.info("-" * 66)
logger.info(f"{'Target':<30} {target:.4f}")
logger.info(f"{'Solver status':<30} {result['status']}")
logger.info(f"{'Solver success':<30} {result.success}")
logger.info(f"{'Solver message':<30} {result.message}")
logger.info(f"{'Solver fun':<30} {result.fun}")
logger.info(f"{'Solver result nit':<30} {result.nit}")
logger.info(f"{'Solver result nfev':<30} {result.nfev}")
logger.info(f"{'Solver result njev':<30} {result.njev}")
logger.info(f"{'Solver result multipliers':<30} {result.multipliers}")

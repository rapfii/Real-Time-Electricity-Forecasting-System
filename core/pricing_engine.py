"""
Deterministic pricing engine.

Converts electricity load (MW) to price ($/MWh) using a fixed quadratic
formula. This is NOT a machine learning model — it is stateless, side-effect
free, deterministic business logic.

Reference: SYSTEM_DESIGN.md Section 10.4
Formula:  P(L) = intercept + linear_coeff * L + quadratic_coeff * L²
Default:  P(L) = 15.0 + 0.005 * L + 0.000002 * L²
"""

import logging

from config.settings import Settings

logger = logging.getLogger(__name__)


class PricingEngine:
    """Deterministic pricing function. NOT a model. Pure business logic.

    The pricing engine is used in two contexts:
    1. Inference path:  PricingEngine.calculate(Load_hat) → predicted price
    2. Ground truth:    PricingEngine.calculate(Load_actual) + ε → actual price
                        (noise ε is added externally, not inside this class)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or Settings()
        self._intercept: float = s.price_intercept
        self._linear: float = s.price_linear_coeff
        self._quadratic: float = s.price_quadratic_coeff

    def calculate(self, load_mw: float) -> float:
        """Compute electricity price from load using the quadratic formula.

        Args:
            load_mw: Electricity load in megawatts.

        Returns:
            Price in $/MWh.
        """
        return (
            self._intercept
            + self._linear * load_mw
            + self._quadratic * (load_mw ** 2)
        )

    def calculate_batch(self, loads_mw: list[float]) -> list[float]:
        """Compute prices for a batch of load values.

        Args:
            loads_mw: List of electricity load values in megawatts.

        Returns:
            List of prices in $/MWh.
        """
        return [self.calculate(load) for load in loads_mw]

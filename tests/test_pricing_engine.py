"""Unit tests for the PricingEngine.

Tests:
  - Correct formula: P(L) = 15.0 + 0.005*L + 0.000002*L²
  - Known values from Appendix A
  - Zero load
  - Batch calculation
  - Stateless / deterministic behavior
"""

import pytest

from config.settings import Settings
from core.pricing_engine import PricingEngine


@pytest.fixture
def engine() -> PricingEngine:
    """Create a PricingEngine with default settings."""
    return PricingEngine(Settings())


class TestPricingEngine:
    """Tests for PricingEngine.calculate()."""

    def test_formula_zero_load(self, engine: PricingEngine) -> None:
        """P(0) = 15.0 (intercept only)."""
        assert engine.calculate(0.0) == pytest.approx(15.0)

    def test_formula_20000_mw(self, engine: PricingEngine) -> None:
        """P(20000) = 15 + 0.005*20000 + 0.000002*20000² = 915.0"""
        expected = 15.0 + 0.005 * 20000 + 0.000002 * (20000 ** 2)
        assert engine.calculate(20000.0) == pytest.approx(expected)
        assert engine.calculate(20000.0) == pytest.approx(915.0)

    def test_formula_30000_mw(self, engine: PricingEngine) -> None:
        """P(30000) = 15 + 150 + 1800 = 1965.0"""
        assert engine.calculate(30000.0) == pytest.approx(1965.0)

    def test_formula_40000_mw(self, engine: PricingEngine) -> None:
        """P(40000) = 15 + 200 + 3200 = 3415.0"""
        assert engine.calculate(40000.0) == pytest.approx(3415.0)

    def test_formula_50000_mw(self, engine: PricingEngine) -> None:
        """P(50000) = 15 + 250 + 5000 = 5265.0"""
        assert engine.calculate(50000.0) == pytest.approx(5265.0)

    def test_deterministic(self, engine: PricingEngine) -> None:
        """Same input always produces same output."""
        v1 = engine.calculate(33000.0)
        v2 = engine.calculate(33000.0)
        assert v1 == v2

    def test_batch_calculation(self, engine: PricingEngine) -> None:
        """Batch matches individual calculations."""
        loads = [20000.0, 30000.0, 40000.0]
        results = engine.calculate_batch(loads)
        for load, result in zip(loads, results):
            assert result == pytest.approx(engine.calculate(load))

    def test_monotonically_increasing(self, engine: PricingEngine) -> None:
        """Higher load should produce higher price (quadratic with positive coefficients)."""
        p1 = engine.calculate(10000.0)
        p2 = engine.calculate(20000.0)
        p3 = engine.calculate(30000.0)
        assert p1 < p2 < p3

    def test_custom_settings(self) -> None:
        """PricingEngine respects custom settings."""
        settings = Settings()
        settings.price_intercept = 10.0
        settings.price_linear_coeff = 0.01
        settings.price_quadratic_coeff = 0.0
        engine = PricingEngine(settings)
        assert engine.calculate(1000.0) == pytest.approx(10.0 + 0.01 * 1000)

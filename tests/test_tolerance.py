"""Tests für die Toleranzintervall-Berechnung."""

import numpy as np
import pytest
from scipy import stats

from src.tolerance_intervals import (
    auto_tolerance_interval,
    min_n_distribution_free,
    distribution_free_ti,
    normal_ti,
    lognormal_ti,
    weibull_ti,
    k_factor,
)


class TestMinN:
    """Mindeststichprobengrößen."""

    def test_95_95_two_sided(self):
        assert min_n_distribution_free(0.95, 0.95, 'two-sided') == 93

    def test_95_95_one_sided(self):
        assert min_n_distribution_free(0.95, 0.95, 'lower') == 59

    def test_99_99_two_sided(self):
        assert min_n_distribution_free(0.99, 0.99, 'two-sided') == 662

    def test_90_90_two_sided(self):
        assert min_n_distribution_free(0.90, 0.90, 'two-sided') == 38


class TestKFactor:
    """k-Faktor-Berechnung."""

    def test_positive(self):
        k = k_factor(20, 0.95, 0.95, 'two-sided')
        assert k > 0

    def test_decreases_with_n(self):
        k20 = k_factor(20, 0.95, 0.95, 'two-sided')
        k100 = k_factor(100, 0.95, 0.95, 'two-sided')
        assert k20 > k100  # Mehr Daten → kleinerer k-Faktor

    def test_one_sided_smaller(self):
        k_two = k_factor(30, 0.95, 0.95, 'two-sided')
        k_one = k_factor(30, 0.95, 0.95, 'upper')
        assert k_one < k_two


class TestDistributionFree:
    """Verteilungsfreies TI."""

    def test_uses_min_max(self):
        """Bei p=0.95 und n=100 muss d=99 sein → ℓ=1, u=100."""
        data = np.array([1, 3, 5, 7, 9] * 20)  # n=100
        result = distribution_free_ti(data, 0.95, 0.95)
        assert result.lower == 1.0
        assert result.upper == 9.0

    def test_inner_order_statistics(self):
        """Bei p=0.90 und n=100 werden innere Ordnungsstatistiken gewählt
        (Meeker Example 5.9: engeres Intervall als Min/Max)."""
        data = np.array([1, 3, 5, 7, 9] * 20)  # n=100, symmetrisch
        result = distribution_free_ti(data, 0.90, 0.95)
        # Muss nicht Min/Max sein
        assert result.confidence >= 0.95
        # Details zeigen die gewählten Ordnungsstatistiken
        assert 'ordnungsstatistiken' in result.details

    def test_meeker_example_5_9(self):
        """Vergleich mit Meeker/Hahn (2017), Example 5.9.
        n=100, p=0.90, conf=0.95, zweiseitig.
        Meeker: [1.66, 37.32] mit l=2, u=98, CPTI=0.9763.
        Wir: optimieren für engstes Intervall bei gleichem d=96."""
        meeker_data = [
            1.49, 1.66, 2.05, 2.24, 2.29, 2.69, 2.77, 2.77, 3.10, 3.23,
            3.28, 3.29, 3.31, 3.36, 3.84, 4.04, 4.09, 4.13, 4.14, 4.16,
            4.57, 4.63, 4.83, 5.06, 5.17, 5.19, 5.89, 5.97, 6.28, 6.38,
            6.51, 6.53, 6.54, 6.55, 6.83, 7.08, 7.28, 7.53, 7.54, 7.68,
            7.81, 7.87, 7.94, 8.43, 8.70, 8.97, 8.98, 9.13, 9.14, 9.22,
            9.24, 9.30, 9.44, 9.69, 9.86, 9.99, 11.28, 11.37, 12.03, 12.32,
            12.93, 13.03, 13.09, 13.43, 13.58, 13.70, 14.17, 14.36, 14.96, 15.89,
            16.57, 16.60, 16.85, 17.18, 17.46, 17.74, 18.40, 18.78, 19.84, 20.45,
            20.89, 22.28, 22.48, 23.66, 24.33, 24.72, 25.46, 25.67, 25.77, 26.64,
            28.28, 28.28, 29.07, 29.16, 31.14, 31.83, 33.24, 37.32, 53.43, 58.11,
        ]
        result = distribution_free_ti(np.array(meeker_data), 0.90, 0.95)
        # Konfidenz muss Meeker-Wert 0.9763 entsprechen (gleicher Span d=96)
        assert abs(result.confidence - 0.9763) < 0.001
        # Unser Intervall muss <= Meeker-Breite sein (Optimierung!)
        meeker_width = 37.32 - 1.66  # = 35.66
        our_width = result.upper - result.lower
        assert our_width <= meeker_width + 0.01

    def test_confidence_met(self):
        data = np.random.default_rng(42).normal(0, 1, size=100)
        result = distribution_free_ti(data, 0.95, 0.95)
        assert result.confidence >= 0.95


class TestNormalTI:
    """Normal-TI."""

    def test_symmetric_around_mean(self):
        data = np.random.default_rng(42).normal(100, 10, size=50)
        result = normal_ti(data, 0.95, 0.95, 'two-sided')
        mean = np.mean(data)
        assert abs((result.upper - mean) - (mean - result.lower)) < 1e-10

    def test_one_sided_upper(self):
        data = np.random.default_rng(42).normal(0, 1, size=30)
        result = normal_ti(data, 0.95, 0.95, 'upper')
        assert result.lower is None
        assert result.upper > np.mean(data)


class TestLognormalTI:
    """Lognormal-TI."""

    def test_positive_bounds(self):
        data = np.random.default_rng(42).lognormal(3, 0.5, size=30)
        result = lognormal_ti(data, 0.95, 0.95)
        assert result.lower > 0
        assert result.upper > 0
        assert result.upper > result.lower

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            lognormal_ti(np.array([-1, 2, 3, 4, 5]), 0.95, 0.95)


class TestWeibullTI:
    """Weibull-TI."""

    def test_positive_bounds(self):
        rng = np.random.default_rng(42)
        data = stats.weibull_min.rvs(1.5, loc=0, scale=100, size=30,
                                      random_state=rng)
        result = weibull_ti(data, 0.95, 0.95)
        assert result.lower > 0
        assert result.upper > 0

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            weibull_ti(np.array([-1, 2, 3, 4, 5]), 0.95, 0.95)


class TestAutoDecision:
    """Automatische Methodenwahl."""

    def test_large_n_uses_distribution_free(self):
        data = np.random.default_rng(42).normal(0, 1, size=100)
        result = auto_tolerance_interval(data, verbose=False)
        assert 'Verteilungsfrei' in result.method

    def test_small_n_normal_uses_normal(self):
        data = np.random.default_rng(42).normal(100, 10, size=20)
        result = auto_tolerance_interval(data, verbose=False)
        assert 'Normal' in result.method

    def test_lognormal_data_detected(self):
        # Höhere Varianz auf log-Skala → deutlicher nicht-normal
        data = np.random.default_rng(42).lognormal(3, 1.0, size=25)
        result = auto_tolerance_interval(data, verbose=False)
        assert 'Lognormal' in result.method

    def test_bimodal_falls_back(self):
        rng = np.random.default_rng(42)
        data = np.concatenate([rng.normal(10, 1, 8), rng.normal(30, 1, 7)])
        result = auto_tolerance_interval(data, verbose=False)
        assert 'FALLBACK' in result.method

    def test_nan_handling(self):
        data = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10] * 10)
        result = auto_tolerance_interval(data, verbose=False)
        assert result.n == 90  # 10 NaN entfernt

    def test_too_few_values_raises(self):
        with pytest.raises(ValueError):
            auto_tolerance_interval([42], verbose=False)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

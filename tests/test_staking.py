import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class KellyFractionTests(unittest.TestCase):
    def test_kelly_fraction_matches_closed_form(self):
        from cs2pickem.staking import kelly_fraction

        # b = 1.0 (even money), p = 0.6 -> f* = (0.6 - 0.4)/1.0 = 0.2.
        self.assertAlmostEqual(kelly_fraction(0.6, 2.0), 0.2, places=9)
        # b = 2.0 (decimal 3.0), p = 0.5 -> f* = (2*0.5 - 0.5)/2 = 0.25.
        self.assertAlmostEqual(kelly_fraction(0.5, 3.0), 0.25, places=9)

    def test_kelly_fraction_clamps_non_positive_edge_to_zero(self):
        from cs2pickem.staking import kelly_fraction

        # Even money, p = 0.4 -> negative edge -> no bet.
        self.assertEqual(kelly_fraction(0.4, 2.0), 0.0)
        # Exactly break-even (b*p == q) -> zero stake.
        self.assertEqual(kelly_fraction(0.5, 2.0), 0.0)

    def test_kelly_fraction_validates_inputs(self):
        from cs2pickem.staking import kelly_fraction

        with self.assertRaises(ValueError):
            kelly_fraction(1.5, 2.0)
        with self.assertRaises(ValueError):
            kelly_fraction(-0.1, 2.0)
        with self.assertRaises(ValueError):
            kelly_fraction(0.6, 1.0)
        with self.assertRaises(ValueError):
            kelly_fraction(0.6, 0.8)


class FractionalKellyTests(unittest.TestCase):
    def test_fractional_kelly_default_is_half_kelly(self):
        from cs2pickem.staking import fractional_kelly, kelly_fraction

        raw = kelly_fraction(0.6, 2.0)
        self.assertAlmostEqual(fractional_kelly(0.6, 2.0), 0.5 * raw, places=9)
        self.assertAlmostEqual(fractional_kelly(0.6, 2.0), 0.10, places=9)

    def test_fractional_kelly_applies_quarter_coefficient(self):
        from cs2pickem.staking import fractional_kelly, kelly_fraction

        raw = kelly_fraction(0.6, 2.0)
        self.assertAlmostEqual(fractional_kelly(0.6, 2.0, fraction=0.25), 0.25 * raw, places=9)

    def test_fractional_kelly_clamps_coefficient_to_half_kelly_ceiling(self):
        from cs2pickem.staking import fractional_kelly, kelly_fraction

        raw = kelly_fraction(0.6, 2.0)
        # Full Kelly request is clamped DOWN to 0.5 (the API discourages > 0.5).
        self.assertAlmostEqual(fractional_kelly(0.6, 2.0, fraction=1.0), 0.5 * raw, places=9)
        # Negative coefficient -> no bet.
        self.assertEqual(fractional_kelly(0.6, 2.0, fraction=-0.3), 0.0)

    def test_fractional_kelly_no_bet_on_negative_edge(self):
        from cs2pickem.staking import fractional_kelly

        self.assertEqual(fractional_kelly(0.4, 2.0), 0.0)


class ExposureCapTests(unittest.TestCase):
    def test_exposure_cap_scales_down_when_total_exceeds_ceiling(self):
        from cs2pickem.staking import portfolio_exposure_cap

        capped = portfolio_exposure_cap([0.2, 0.3, 0.1], max_total_exposure=0.3)
        self.assertAlmostEqual(sum(capped), 0.3, places=9)
        # Proportional shrink preserves the relative sizing.
        self.assertAlmostEqual(capped[0] / capped[1], 0.2 / 0.3, places=9)
        self.assertAlmostEqual(capped[0], 0.2 * (0.3 / 0.6), places=9)

    def test_exposure_cap_is_a_noop_below_ceiling(self):
        from cs2pickem.staking import portfolio_exposure_cap

        stakes = [0.05, 0.04, 0.03]
        self.assertEqual(portfolio_exposure_cap(stakes, max_total_exposure=0.25), stakes)

    def test_exposure_cap_supports_mapping_and_floors_negatives(self):
        from cs2pickem.staking import portfolio_exposure_cap

        capped = portfolio_exposure_cap({"a": 0.4, "b": 0.4, "c": -0.1}, max_total_exposure=0.4)
        self.assertEqual(capped["c"], 0.0)
        self.assertAlmostEqual(capped["a"] + capped["b"] + capped["c"], 0.4, places=9)
        self.assertAlmostEqual(capped["a"], capped["b"], places=9)

    def test_exposure_cap_default_ceiling_is_documented_conservative_value(self):
        from cs2pickem.staking import DEFAULT_MAX_TOTAL_EXPOSURE, portfolio_exposure_cap

        self.assertLessEqual(DEFAULT_MAX_TOTAL_EXPOSURE, 0.5)
        capped = portfolio_exposure_cap([0.3, 0.3, 0.3])
        self.assertAlmostEqual(sum(capped), DEFAULT_MAX_TOTAL_EXPOSURE, places=9)


class KellyReportTests(unittest.TestCase):
    def test_kelly_report_exposes_edge_and_fractional_stake(self):
        from cs2pickem.staking import kelly_report

        report = kelly_report(0.6, 2.0)
        self.assertAlmostEqual(report["edge"], 0.2, places=9)
        self.assertAlmostEqual(report["raw_kelly"], 0.2, places=9)
        self.assertAlmostEqual(report["fraction"], 0.5, places=9)
        self.assertAlmostEqual(report["fractional_kelly"], 0.10, places=9)


if __name__ == "__main__":
    unittest.main()

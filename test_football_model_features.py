import unittest

from football_model_features import add_venue_comparison, enhanced_rolling_features


class FootballModelFeatureTests(unittest.TestCase):
    def test_recent_results_have_more_weight(self):
        recent_wins = enhanced_rolling_features(
            [(3, 2, 0)] * 5 + [(0, 0, 2)] * 5, "form"
        )
        old_wins = enhanced_rolling_features(
            [(0, 0, 2)] * 5 + [(3, 2, 0)] * 5, "form"
        )
        self.assertGreater(recent_wins["form_weighted_ppg"], old_wins["form_weighted_ppg"])

    def test_small_sample_is_shrunk_to_prior(self):
        result = enhanced_rolling_features([(3, 2, 0)], "form")
        self.assertGreater(result["form_shrunk_ppg"], 1.35)
        self.assertLess(result["form_shrunk_ppg"], 3.0)
        self.assertEqual(result["form_sample_reliability"], .2)

    def test_strong_opponents_increase_adjusted_form(self):
        strong = enhanced_rolling_features([(3, 1, 0, 2.2)] * 5, "form")
        weak = enhanced_rolling_features([(3, 1, 0, .6)] * 5, "form")
        self.assertGreater(strong["form_strength_adjusted_ppg"], weak["form_strength_adjusted_ppg"])

    def test_venue_comparison_can_favor_visitor(self):
        features = {
            "form_home_10_strength_adjusted_ppg": 1.5,
            "form_away_10_strength_adjusted_ppg": 1.6,
            "form_home_casa_10_strength_adjusted_ppg": 1.2,
            "form_away_fora_10_strength_adjusted_ppg": 2.0,
            "form_home_casa_10_sample_reliability": 1.0,
            "form_away_fora_10_sample_reliability": 1.0,
        }
        add_venue_comparison(features)
        self.assertLess(features["context_mando_ajustado_gap"], 0)


if __name__ == "__main__":
    unittest.main()

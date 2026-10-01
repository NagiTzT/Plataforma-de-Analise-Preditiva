import unittest

from competition_context import competition_flags


class CompetitionContextTests(unittest.TestCase):
    def test_cups_are_not_treated_as_regular_leagues(self):
        flags = competition_flags("England EFL Cup")
        self.assertEqual(flags["is_cup"], 1.0)
        self.assertEqual(flags["is_knockout"], 1.0)

    def test_group_stage_cup_is_not_automatically_knockout(self):
        flags = competition_flags("World Cup Group A")
        self.assertEqual(flags["is_cup"], 1.0)
        self.assertEqual(flags["is_knockout"], 0.0)

    def test_regular_league_has_low_volatility(self):
        flags = competition_flags("Japan J1 League")
        self.assertEqual(flags["is_cup"], 0.0)
        self.assertEqual(flags["is_knockout"], 0.0)
        self.assertLess(flags["competition_volatility"], 0.15)

    def test_youth_and_women_flags(self):
        self.assertEqual(competition_flags("Premier League U21")["is_youth_or_reserve"], 1.0)
        self.assertEqual(competition_flags("Serie A Women")["is_women"], 1.0)

    def test_development_and_lower_tier_leagues_are_volatile(self):
        next_pro = competition_flags("USA MLS Next Pro League")
        kolmonen = competition_flags("Finland Kolmonen")
        self.assertEqual(next_pro["is_youth_or_reserve"], 1.0)
        self.assertEqual(kolmonen["is_lower_tier"], 1.0)
        self.assertGreater(next_pro["competition_volatility"], .20)
        self.assertGreater(kolmonen["competition_volatility"], .15)

    def test_friendlies_have_their_own_calibration_family(self):
        flags = competition_flags("International Friendly Games")
        self.assertEqual(flags["is_friendly"], 1.0)
        self.assertEqual(flags["competition_family_friendly"], 1.0)
        self.assertEqual(flags["competition_family_league"], 0.0)

    def test_competition_family_is_mutually_exclusive(self):
        for name in ("Japan J1 League", "England EFL Cup",
                     "World Cup Qualifier", "World Cup Group A",
                     "World Cup Qualification Group A", "Friendly Cup Group A"):
            flags = competition_flags(name)
            family = [value for key, value in flags.items()
                      if key.startswith("competition_family_")]
            self.assertEqual(sum(family), 1.0, name)

    def test_icelandic_umspil_is_an_explicit_knockout_stage(self):
        # KSÍ 2026 Lengjudeild rules 23.1.8.2: promotion semi-finals and final.
        flags = competition_flags("Lengjudeild karla 2026 - Umspil")
        self.assertEqual(flags["is_knockout"], 1.0)
        self.assertEqual(flags["competition_family_knockout"], 1.0)
        self.assertEqual(flags["is_cup"], 0.0)

    def test_explicit_round_can_be_supplied_without_guessing_from_date(self):
        plain = competition_flags("Iceland 1. deild")
        stage = competition_flags("Iceland 1. deild", stage_name="Promotion Play-offs")
        self.assertEqual(plain["is_knockout"], 0.0)
        self.assertEqual(stage["is_knockout"], 1.0)

    def test_current_elimination_round_overrides_old_group_label(self):
        flags = competition_flags("World Cup Group Stage", stage_name="Quarter-finals")
        self.assertEqual(flags["is_knockout"], 1.0)
        self.assertEqual(flags["competition_family_knockout"], 1.0)
        self.assertEqual(flags["competition_family_cup_group"], 0.0)

    def test_playoff_spellings_and_rounds(self):
        for stage in ("Playoff", "Playoffs", "Play-off", "Play-offs",
                      "Semi-finals", "Quarter-finals", "Round of 16"):
            self.assertEqual(competition_flags("League", stage)["is_knockout"], 1.0, stage)

    def test_promotion_and_relegation_groups_are_not_automatically_knockout(self):
        for name in ("Premier League Relegation Group", "Promotion Play-off Group",
                     "Final Group", "League Promotion", "League Relegation",
                     "World Cup Final Group Stage"):
            self.assertEqual(competition_flags(name)["is_knockout"], 0.0, name)

    def test_knockout_tokens_do_not_match_unrelated_word_substrings(self):
        self.assertEqual(competition_flags("League Finalization")["is_knockout"], 0.0)

    def test_final_phase_alone_does_not_establish_elimination_format(self):
        for stage in ("Final stage", "Final phase", "Final round", "Fase final"):
            self.assertEqual(competition_flags("League", stage)["is_knockout"], 0.0, stage)


if __name__ == "__main__":
    unittest.main()

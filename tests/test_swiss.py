import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class SwissPairingTests(unittest.TestCase):
    def test_opening_pairings_use_snake_seed_order(self):
        from cs2pickem.swiss import TeamSeed, _opening_pairings

        teams = [TeamSeed(f"Team{seed}", seed) for seed in range(1, 9)]

        pairings = _opening_pairings(teams)

        self.assertEqual(
            [(left.seed, right.seed) for left, right in pairings],
            [(1, 8), (2, 7), (3, 6), (4, 5)],
        )

    def test_active_pairings_stay_in_score_groups_with_snake_order_and_no_rematch(self):
        from cs2pickem.swiss import TeamSeed, TeamState, _pair_active

        states = {
            f"Team{seed}": TeamState(team=TeamSeed(f"Team{seed}", seed))
            for seed in range(1, 9)
        }
        for seed in (1, 4, 5, 8):
            states[f"Team{seed}"].wins = 1
        for seed in (2, 3, 6, 7):
            states[f"Team{seed}"].losses = 1
        states["Team1"].opponents.add("Team8")
        states["Team8"].opponents.add("Team1")

        pairings = _pair_active(states)

        self.assertEqual(
            [(left.seed, right.seed) for left, right in pairings],
            [(1, 5), (4, 8), (2, 7), (3, 6)],
        )


if __name__ == "__main__":
    unittest.main()

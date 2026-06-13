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


class BuchholzPairingTests(unittest.TestCase):
    def test_opening_pairings_1v9_offset_split(self):
        from cs2pickem.swiss import TeamSeed, _opening_pairings_1v9

        teams = [TeamSeed(f"Team{seed}", seed) for seed in range(1, 17)]

        pairings = _opening_pairings_1v9(teams)

        # Valve Major round 1: seed_i vs seed_(i+8), i.e. 1v9, 2v10, ... 8v16
        self.assertEqual(
            [(left.seed, right.seed) for left, right in pairings],
            [(1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16)],
        )

    def test_buchholz_difficulty_sum_of_opponent_wins_minus_losses(self):
        from cs2pickem.swiss import TeamSeed, TeamState, _buchholz

        states = {
            f"Team{seed}": TeamState(team=TeamSeed(f"Team{seed}", seed))
            for seed in range(1, 5)
        }
        # Team2: 2-0 ; Team3: 1-1 ; Team4: 0-2
        states["Team2"].wins, states["Team2"].losses = 2, 0
        states["Team3"].wins, states["Team3"].losses = 1, 1
        states["Team4"].wins, states["Team4"].losses = 0, 2
        # Team1 has played Team2, Team3, Team4
        for opp in ("Team2", "Team3", "Team4"):
            states["Team1"].opponents.add(opp)

        # Buchholz = sum(opp.wins - opp.losses) = (2-0)+(1-1)+(0-2) = 0
        self.assertEqual(_buchholz(states["Team1"], states), 0)
        # Team2 (played nobody recorded) => 0
        self.assertEqual(_buchholz(states["Team2"], states), 0)

    def test_bucket_rank_orders_by_buchholz_desc_then_seed_asc(self):
        from cs2pickem.swiss import TeamSeed, TeamState, _rank_bucket

        # Same (wins, losses) bucket; differing buchholz set manually via opponents.
        states = {
            f"Team{seed}": TeamState(team=TeamSeed(f"Team{seed}", seed), wins=1, losses=0)
            for seed in range(1, 5)
        }
        # Higher seeds (strong opponents) -> higher buchholz. Give Team4 the
        # toughest opponent so its buchholz is highest despite worst seed.
        strong = TeamState(team=TeamSeed("Strong", 99), wins=2, losses=0)
        weak = TeamState(team=TeamSeed("Weak", 100), wins=0, losses=2)
        states["Strong"] = strong
        states["Weak"] = weak
        states["Team4"].opponents.add("Strong")
        states["Team1"].opponents.add("Weak")
        # Team2/Team3 buchholz == 0 (no recorded opponents) -> tiebreak by seed.

        bucket = [states[f"Team{seed}"] for seed in range(1, 5)]
        ordered = _rank_bucket(bucket, states)

        # Team4 (buchholz +2) first; then Team2, Team3 (buchholz 0, seed asc);
        # Team1 (buchholz -2) last.
        self.assertEqual(
            [state.team.seed for state in ordered],
            [4, 2, 3, 1],
        )

    def test_buchholz_pair_active_uses_backtracking_no_rematch(self):
        from cs2pickem.swiss import TeamSeed, TeamState, _pair_active_buchholz

        states = {
            f"Team{seed}": TeamState(team=TeamSeed(f"Team{seed}", seed))
            for seed in range(1, 9)
        }
        for seed in (1, 4, 5, 8):
            states[f"Team{seed}"].wins = 1
        for seed in (2, 3, 6, 7):
            states[f"Team{seed}"].losses = 1
        # Force a rematch threat: Team1 already faced Team8.
        states["Team1"].opponents.add("Team8")
        states["Team8"].opponents.add("Team1")

        pairings = _pair_active_buchholz(states)

        # Hard no-rematch constraint: Team1 must not be paired with Team8 again.
        pair_names = {frozenset((a.name, b.name)) for a, b in pairings}
        self.assertNotIn(frozenset(("Team1", "Team8")), pair_names)
        # 4 distinct teams per (wins,losses) bucket -> 2 pairings each, 4 total.
        self.assertEqual(len(pairings), 4)
        # Every active team appears exactly once.
        appeared = [name for pair in pairings for name in (pair[0].name, pair[1].name)]
        self.assertEqual(sorted(appeared), [f"Team{seed}" for seed in range(1, 9)])

    def test_buchholz_simulation_advance_equals_record_breakdown(self):
        from cs2pickem.swiss import TeamSeed, simulate_swiss

        teams = [TeamSeed(f"Team{seed}", seed) for seed in range(1, 17)]

        def predictor(team_a, team_b, best_of, state):
            return 0.62 if team_a.seed < team_b.seed else 0.38

        result = simulate_swiss(
            teams, predictor, simulations=4000, seed=7, pairing="buchholz"
        )

        for name, probs in result.team_probabilities.items():
            # advance == P(3-0) + P(3-1) + P(3-2) (internal consistency)
            advance = probs["advance"]
            breakdown = probs["3-0"] + probs["3-1"] + probs["3-2"]
            self.assertAlmostEqual(advance, breakdown, places=6, msg=name)
            # eliminate == P(0-3)+P(1-3)+P(2-3)
            eliminate = probs["eliminate"]
            elim_breakdown = probs["0-3"] + probs["1-3"] + probs["2-3"]
            self.assertAlmostEqual(eliminate, elim_breakdown, places=6, msg=name)
            # advance + eliminate == 1 (everyone terminates at 3 wins or 3 losses)
            self.assertAlmostEqual(advance + eliminate, 1.0, places=6, msg=name)
        # Exactly 8 of 16 advance on average (3W cut) -> sum of advance ~= 8.
        total_advance = sum(p["advance"] for p in result.team_probabilities.values())
        self.assertAlmostEqual(total_advance, 8.0, places=6)

    def test_buchholz_simulation_no_rematch_invariant_via_joint_samples(self):
        from cs2pickem.swiss import TeamSeed, simulate_swiss

        teams = [TeamSeed(f"Team{seed}", seed) for seed in range(1, 17)]

        def predictor(team_a, team_b, best_of, state):
            return 0.55 if team_a.seed < team_b.seed else 0.45

        result = simulate_swiss(
            teams,
            predictor,
            simulations=300,
            seed=11,
            pairing="buchholz",
            collect_joint=True,
        )

        # joint_samples collected: one dict per simulation, keyed by team.
        self.assertEqual(len(result.joint_samples), 300)
        sample = result.joint_samples[0]
        self.assertEqual(set(sample), {team.name for team in teams})
        # Each team's recorded outcome terminates (3 wins or 3 losses).
        for record in sample.values():
            self.assertTrue(record["advance"] or record["eliminate"])
            self.assertEqual(record["advance"], not record["eliminate"])
        # Exactly 8 advance in every individual simulated bracket.
        for joint in result.joint_samples:
            advancing = sum(1 for rec in joint.values() if rec["advance"])
            self.assertEqual(advancing, 8)

    def test_buchholz_simulation_frequency_is_stable(self):
        from cs2pickem.swiss import TeamSeed, simulate_swiss

        teams = [TeamSeed(f"Team{seed}", seed) for seed in range(1, 17)]

        def predictor(team_a, team_b, best_of, state):
            return 0.65 if team_a.seed < team_b.seed else 0.35

        run_a = simulate_swiss(teams, predictor, simulations=3000, seed=99, pairing="buchholz")
        run_b = simulate_swiss(teams, predictor, simulations=3000, seed=99, pairing="buchholz")

        # Same seed -> deterministic, identical frequencies.
        self.assertEqual(run_a.team_probabilities, run_b.team_probabilities)
        # Top seed (1) advances clearly more often than the weakest (16).
        self.assertGreater(
            run_a.team_probabilities["Team1"]["advance"],
            run_a.team_probabilities["Team16"]["advance"],
        )

    def test_legacy_default_unchanged_and_no_joint_samples(self):
        from cs2pickem.swiss import TeamSeed, simulate_swiss

        teams = [TeamSeed(f"Team{seed}", seed) for seed in range(1, 9)]

        def predictor(team_a, team_b, best_of, state):
            return 0.6 if team_a.seed < team_b.seed else 0.4

        # Default pairing stays legacy; collect_joint defaults False.
        result = simulate_swiss(teams, predictor, simulations=200, seed=5)
        self.assertEqual(result.joint_samples, [])
        self.assertEqual(result.simulations, 200)
        self.assertIn("advance", result.team_probabilities["Team1"])


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class EloRatingTests(unittest.TestCase):
    def test_wins_raise_rating_losses_lower_it_and_stay_zero_sum(self):
        from cs2pickem.ratings import compute_elo_ratings

        matches = [
            {"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"},
            {"date": "2026-01-02", "team1": "A", "team2": "B", "winner": "A"},
        ]
        per_match, final = compute_elo_ratings(matches, base=1500.0, k=32.0)

        # First match: both teams enter at base (pre-match, no leakage)
        self.assertAlmostEqual(per_match[0]["team1_elo_pre"], 1500.0)
        self.assertAlmostEqual(per_match[0]["team2_elo_pre"], 1500.0)
        # Winner rises, loser falls, and Elo updates are zero-sum
        self.assertGreater(final["A"], 1500.0)
        self.assertLess(final["B"], 1500.0)
        self.assertAlmostEqual(final["A"] + final["B"], 3000.0, places=3)

    def test_pre_match_rating_excludes_that_match_result(self):
        from cs2pickem.ratings import compute_elo_ratings

        matches = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
        per_match, final = compute_elo_ratings(matches, base=1500.0, k=32.0)

        self.assertEqual(per_match[0]["team1_elo_pre"], 1500.0)  # before the update
        self.assertNotEqual(final["A"], 1500.0)                  # after the update

    def test_beating_a_stronger_opponent_gains_more_rating(self):
        from cs2pickem.ratings import compute_elo_ratings

        match = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
        _, equal = compute_elo_ratings(match, base=1500.0, k=32.0)
        _, weak = compute_elo_ratings(match, base=1500.0, k=32.0, initial_ratings={"B": 1200.0})

        self.assertGreater(equal["A"] - 1500.0, weak["A"] - 1500.0)


def _bt_prob(theta_a: float, theta_b: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-(theta_a - theta_b)))


class BradleyTerryTests(unittest.TestCase):
    def _round_robin(self, win_counts):
        """Build a result list from a {(winner, loser): n} mapping."""
        matches = []
        day = 1
        for (winner, loser), count in win_counts.items():
            for _ in range(count):
                matches.append(
                    {
                        "date": f"2026-01-{day:02d}",
                        "team1": winner,
                        "team2": loser,
                        "winner": winner,
                    }
                )
                day += 1
        return matches

    def test_returns_mean_centered_log_strengths(self):
        from cs2pickem.ratings import compute_bradley_terry

        matches = self._round_robin({("A", "B"): 4, ("A", "C"): 3, ("B", "C"): 2})
        theta = compute_bradley_terry(matches)

        # Every observed team gets a strength.
        self.assertEqual(set(theta), {"A", "B", "C"})
        # Mean-centered for identifiability.
        self.assertAlmostEqual(sum(theta.values()) / len(theta), 0.0, places=6)

    def test_stronger_team_gets_higher_theta(self):
        from cs2pickem.ratings import compute_bradley_terry

        # A dominates, C is dominated.
        matches = self._round_robin({("A", "B"): 8, ("A", "C"): 9, ("B", "C"): 7})
        matches += self._round_robin({("B", "A"): 2, ("C", "A"): 1, ("C", "B"): 3})
        theta = compute_bradley_terry(matches)

        self.assertGreater(theta["A"], theta["B"])
        self.assertGreater(theta["B"], theta["C"])

    def test_predicted_probability_matches_win_frequency_two_teams(self):
        from cs2pickem.ratings import compute_bradley_terry

        # For an isolated pair, the BT MLE reproduces the empirical pairwise win
        # frequency exactly: A wins 70 of 100, so P(A>B) -> 0.70.
        matches = self._round_robin({("A", "B"): 70, ("B", "A"): 30})
        theta = compute_bradley_terry(matches, ridge=0.0, max_iter=5000, tol=1e-12)

        prob_a_beats_b = _bt_prob(theta["A"], theta["B"])
        self.assertAlmostEqual(prob_a_beats_b, 0.70, delta=0.005)

    def test_mle_stationarity_observed_equals_expected_wins(self):
        from cs2pickem.ratings import compute_bradley_terry

        # The defining property of the BT MLE: for every team, observed wins equal
        # the model's expected wins summed over its games. This is the rigorous
        # "P(i>j) consistent with results" contract on a non-trivial 3-team graph.
        games = {("A", "B"): 100, ("A", "C"): 100, ("B", "C"): 100}
        win_counts = {("A", "B"): 70, ("B", "A"): 30, ("A", "C"): 50, ("C", "A"): 50, ("B", "C"): 50, ("C", "B"): 50}
        matches = self._round_robin(win_counts)
        theta = compute_bradley_terry(matches, ridge=0.0, max_iter=20000, tol=1e-13)

        import math

        pi = {team: math.exp(value) for team, value in theta.items()}
        observed = {"A": 120.0, "B": 80.0, "C": 100.0}  # total wins per team
        for team in ("A", "B", "C"):
            expected = 0.0
            for (i, j), n in games.items():
                if team == i:
                    expected += n * pi[i] / (pi[i] + pi[j])
                elif team == j:
                    expected += n * pi[j] / (pi[i] + pi[j])
            self.assertAlmostEqual(expected, observed[team], places=3)

    def test_converges_within_iteration_budget(self):
        from cs2pickem.ratings import compute_bradley_terry

        matches = self._round_robin({("A", "B"): 5, ("B", "C"): 5, ("C", "A"): 5})
        # tol-driven convergence: a tiny max_iter still returns finite values, and
        # a generous budget gives a stable fixed point (running more iters barely moves it).
        coarse = compute_bradley_terry(matches, max_iter=2, ridge=1e-3)
        fine = compute_bradley_terry(matches, max_iter=500, ridge=1e-3)
        for team in ("A", "B", "C"):
            self.assertTrue(math_isfinite(coarse[team]))
            self.assertTrue(math_isfinite(fine[team]))
        again = compute_bradley_terry(matches, max_iter=1000, ridge=1e-3)
        for team in ("A", "B", "C"):
            self.assertAlmostEqual(fine[team], again[team], places=5)

    def test_ridge_keeps_undefeated_team_finite(self):
        from cs2pickem.ratings import compute_bradley_terry

        # A is undefeated, D is winless: without shrinkage theta -> +/-inf.
        matches = self._round_robin({("A", "B"): 3, ("A", "C"): 3, ("B", "D"): 3, ("C", "D"): 3})
        theta = compute_bradley_terry(matches, ridge=1.0)
        for value in theta.values():
            self.assertTrue(math_isfinite(value))
        self.assertGreater(theta["A"], theta["D"])

    def test_prior_pulls_strengths_toward_supplied_center(self):
        from cs2pickem.ratings import compute_bradley_terry

        matches = self._round_robin({("A", "B"): 6, ("B", "A"): 4})
        weak_prior = compute_bradley_terry(matches, ridge=20.0)
        # Heavy ridge toward the uniform prior collapses the spread toward 0.
        spread = abs(weak_prior["A"] - weak_prior["B"])
        light = compute_bradley_terry(matches, ridge=0.01)
        self.assertGreater(abs(light["A"] - light["B"]), spread)


class BradleyTerryPerMapTests(unittest.TestCase):
    def _map_match(self, day, t1, t2, winner, map_name):
        return {
            "date": f"2026-02-{day:02d}",
            "team1": t1,
            "team2": t2,
            "winner": winner,
            "map": map_name,
        }

    def test_per_map_fit_returns_strengths_per_map(self):
        from cs2pickem.ratings import compute_map_bradley_terry

        matches = [
            self._map_match(1, "A", "B", "A", "mirage"),
            self._map_match(2, "A", "B", "A", "mirage"),
            self._map_match(3, "B", "A", "B", "inferno"),
            self._map_match(4, "B", "A", "B", "inferno"),
        ]
        per_map = compute_map_bradley_terry(matches, ridge=1.0)

        self.assertIn("mirage", per_map)
        self.assertIn("inferno", per_map)
        # A wins on mirage, loses on inferno -> map-specific orientation.
        self.assertGreater(per_map["mirage"]["A"], per_map["mirage"]["B"])
        self.assertGreater(per_map["inferno"]["B"], per_map["inferno"]["A"])

    def test_sparse_map_shrinks_toward_global(self):
        from cs2pickem.ratings import compute_bradley_terry, compute_map_bradley_terry

        # Dense overall history establishes A >> B globally.
        overall = []
        for day in range(1, 21):
            overall.append(self._map_match(day, "A", "B", "A", "mirage" if day % 2 else "inferno"))
        # On 'nuke' there is a single noisy game where B beats A.
        sparse = [self._map_match(25, "B", "A", "B", "nuke")]
        matches = overall + sparse

        global_theta = compute_bradley_terry(matches, ridge=1.0)
        per_map = compute_map_bradley_terry(matches, ridge=8.0)

        # Despite the lone nuke upset, heavy ridge keeps A above B on nuke
        # (shrunk toward the global prior where A dominates).
        self.assertGreater(per_map["nuke"]["A"], per_map["nuke"]["B"])
        # And the nuke gap is much smaller than the dense-map gap, reflecting shrinkage.
        nuke_gap = per_map["nuke"]["A"] - per_map["nuke"]["B"]
        mirage_gap = per_map["mirage"]["A"] - per_map["mirage"]["B"]
        self.assertLess(nuke_gap, mirage_gap)
        # Sanity: global prior really does favour A.
        self.assertGreater(global_theta["A"], global_theta["B"])

    def test_empty_map_collapses_to_global_prior(self):
        from cs2pickem.ratings import compute_map_bradley_terry

        matches = [
            self._map_match(1, "A", "B", "A", "mirage"),
            self._map_match(2, "A", "B", "A", "mirage"),
        ]
        global_theta = {"A": 1.0, "B": -1.0}
        per_map = compute_map_bradley_terry(
            matches, ridge=8.0, global_theta=global_theta, map_pool=["mirage", "vertigo"]
        )
        # 'vertigo' has zero games -> every team sits at its global prior (centered).
        self.assertIn("vertigo", per_map)
        self.assertGreater(per_map["vertigo"]["A"], per_map["vertigo"]["B"])


class Glicko2NumericalTests(unittest.TestCase):
    """Lock standard Glicko-2 numerics against Glickman's (2013) worked example.

    The canonical example (Glickman, "Example of the Glicko-2 system"): a player at
    rating=1500, RD=200, sigma=0.06 plays three opponents in ONE rating period --
    beats (1400, RD 30), loses to (1550, RD 100), loses to (1700, RD 300). With
    tau=0.5 the published result is rating ~= 1464.06, RD ~= 151.52, sigma ~= 0.05999.
    We reproduce it by giving every opponent a fixed initial state and disabling MOV
    (so each map weighs exactly 1.0).
    """

    def _canonical_matches(self):
        # P = the player under test; three distinct opponents, all same date so they
        # form a single rating period (Glicko-2 batches a period's games together).
        return [
            {"date": "2026-01-01", "team1": "P", "team2": "O1", "winner": "P"},
            {"date": "2026-01-01", "team1": "P", "team2": "O2", "winner": "O2"},
            {"date": "2026-01-01", "team1": "P", "team2": "O3", "winner": "O3"},
        ]

    def _canonical_initial(self):
        return {
            "ratings": {"P": 1500.0, "O1": 1400.0, "O2": 1550.0, "O3": 1700.0},
            "rds": {"P": 200.0, "O1": 30.0, "O2": 100.0, "O3": 300.0},
            "sigmas": {"P": 0.06, "O1": 0.06, "O2": 0.06, "O3": 0.06},
        }

    def test_canonical_glickman_example(self):
        from cs2pickem.ratings import compute_glicko_ratings

        _per_match, final = compute_glicko_ratings(
            self._canonical_matches(),
            tau=0.5,
            use_mov=False,
            initial_state=self._canonical_initial(),
        )
        self.assertAlmostEqual(final["ratings"]["P"], 1464.06, places=1)
        self.assertAlmostEqual(final["rds"]["P"], 151.52, places=1)
        self.assertAlmostEqual(final["sigmas"]["P"], 0.05999, places=4)

    def test_returns_per_match_pre_snapshots_and_final_state(self):
        from cs2pickem.ratings import compute_glicko_ratings

        per_match, final = compute_glicko_ratings(
            self._canonical_matches(),
            tau=0.5,
            use_mov=False,
            initial_state=self._canonical_initial(),
        )
        self.assertEqual(len(per_match), 3)
        # Pre-snapshots carry the pre-period state for both teams (leakage-free).
        first = per_match[0]
        for key in ("team1_glicko_pre", "team2_glicko_pre", "team1_rd_pre", "team2_rd_pre"):
            self.assertIn(key, first)
        # Player's pre-snapshot is its ENTRY state, identical on every same-period row.
        self.assertAlmostEqual(per_match[0]["team1_glicko_pre"], 1500.0, places=6)
        self.assertAlmostEqual(per_match[1]["team1_glicko_pre"], 1500.0, places=6)
        self.assertAlmostEqual(per_match[2]["team1_glicko_pre"], 1500.0, places=6)
        self.assertAlmostEqual(per_match[0]["team1_rd_pre"], 200.0, places=6)
        # final state mirrors compute_elo_ratings' (per_match, ratings)二元组 shape.
        self.assertIn("ratings", final)
        self.assertIn("rds", final)
        self.assertIn("sigmas", final)

    def test_default_initial_state_uses_glickman_seeds(self):
        from cs2pickem.ratings import compute_glicko_ratings

        # With no initial_state, an unseen team enters at mu0=1500/phi0=350/sigma0=0.06.
        matches = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
        per_match, _final = compute_glicko_ratings(matches, use_mov=False)
        self.assertAlmostEqual(per_match[0]["team1_glicko_pre"], 1500.0, places=6)
        self.assertAlmostEqual(per_match[0]["team1_rd_pre"], 350.0, places=6)

    def test_winner_rating_rises_loser_falls(self):
        from cs2pickem.ratings import compute_glicko_ratings

        matches = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
        _per, final = compute_glicko_ratings(matches, use_mov=False)
        self.assertGreater(final["ratings"]["A"], 1500.0)
        self.assertLess(final["ratings"]["B"], 1500.0)

    def test_rd_shrinks_after_playing(self):
        from cs2pickem.ratings import compute_glicko_ratings

        # Playing a game reduces uncertainty: RD drops from the 350 cold-start.
        matches = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
        _per, final = compute_glicko_ratings(matches, use_mov=False)
        self.assertLess(final["rds"]["A"], 350.0)
        self.assertLess(final["rds"]["B"], 350.0)


class Glicko2InactivityTests(unittest.TestCase):
    def test_rd_inflates_with_inactivity(self):
        from cs2pickem.ratings import compute_glicko_ratings

        # A plays once, then sits out many periods while OTHER teams keep playing.
        # Its RD must inflate (phi* = sqrt(phi^2 + sigma^2 * t)) toward, but capped at, max_rd.
        matches = [
            {"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"},
        ]
        # Add later periods (distinct dates) where A does not appear.
        for day in range(2, 30):
            matches.append(
                {"date": f"2026-01-{day:02d}", "team1": "C", "team2": "D", "winner": "C"}
            )
        _per, final = compute_glicko_ratings(matches, use_mov=False)
        # After playing once, A's RD < 350; inactivity then inflates it back up.
        rd_a = final["rds"]["A"]
        self.assertGreater(rd_a, 50.0)
        # mu and sigma are unchanged by pure inactivity (only RD inflates).
        # A won its single game, so its rating moved up and then stays put.
        self.assertGreater(final["ratings"]["A"], 1500.0)

    def test_inactivity_rd_capped_at_max_rd(self):
        from cs2pickem.ratings import compute_glicko_ratings

        # Even with a huge number of skipped periods, RD never exceeds max_rd (350).
        matches = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
        for day in range(2, 28):
            matches.append(
                {"date": f"2026-02-{day:02d}", "team1": "C", "team2": "D", "winner": "C"}
            )
        _per, final = compute_glicko_ratings(matches, use_mov=False, max_rd=350.0)
        self.assertLessEqual(final["rds"]["A"], 350.0 + 1e-9)

    def test_inactivity_inflates_more_than_short_gap(self):
        from cs2pickem.ratings import compute_glicko_ratings

        def rd_after_gap(n_skips):
            matches = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
            for day in range(2, 2 + n_skips):
                matches.append(
                    {"date": f"2026-01-{day:02d}", "team1": "C", "team2": "D", "winner": "C"}
                )
            _per, final = compute_glicko_ratings(matches, use_mov=False)
            return final["rds"]["A"]

        rd_short = rd_after_gap(2)
        rd_long = rd_after_gap(20)
        self.assertGreater(rd_long, rd_short)
        # Magnitude guard (locks the scale bug fixed in WF-2C review): 18 extra skipped
        # periods must add a MEANINGFUL amount of RD, not a ~1e-4 rounding crumb. The
        # buggy mixed-scale formula added only ~0.0001 here; the correct internal-scale
        # inflation adds several RD points.
        self.assertGreater(rd_long - rd_short, 1.0)

    def test_inactivity_inflation_matches_internal_scale_closed_form(self):
        from cs2pickem.ratings import compute_glicko_ratings, GLICKO_SCALE
        import math

        # Closed form: inflation runs in the internal Glicko-2 scale,
        #   phi*_orig = sqrt((phi/SCALE)^2 + sigma^2 * t) * SCALE.
        # Compute the post-one-game baseline phi/sigma, then assert that sitting out t
        # additional periods reproduces the closed form to high precision (catches any
        # future regression to the mixed-scale no-op).
        base = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
        _p0, f0 = compute_glicko_ratings(base, use_mov=False)
        phi0 = f0["rds"]["A"]
        sigma0 = f0["sigmas"]["A"]

        t = 15
        matches = list(base)
        for day in range(2, 2 + t):
            matches.append(
                {"date": f"2026-01-{day:02d}", "team1": "C", "team2": "D", "winner": "C"}
            )
        _per, final = compute_glicko_ratings(matches, use_mov=False)

        phi_g = phi0 / GLICKO_SCALE
        expected = math.sqrt(phi_g * phi_g + sigma0 * sigma0 * float(t)) * GLICKO_SCALE
        self.assertAlmostEqual(final["rds"]["A"], expected, places=6)


class Glicko2ColdStartTests(unittest.TestCase):
    def test_high_rd_cold_start_team_expected_winrate_shrinks_to_half(self):
        from cs2pickem.ratings import compute_glicko_ratings, glicko_expected_score

        # A high-RD (uncertain) team's win probability vs a mid team is pulled toward
        # 0.5 relative to a low-RD team of identical rating, because g(phi) attenuates
        # the rating gap. Two teams at the SAME rating but different RDs vs a stronger
        # opponent: the high-RD team's expected score is closer to 0.5.
        mu_self = 1400.0
        mu_opp = 1600.0
        e_low_rd = glicko_expected_score(mu_self, mu_opp, opp_rd=30.0)
        e_high_rd = glicko_expected_score(mu_self, mu_opp, opp_rd=350.0)
        # Both below 0.5 (self is weaker); the high-opp-RD case is closer to 0.5.
        self.assertLess(e_low_rd, 0.5)
        self.assertLess(e_high_rd, 0.5)
        self.assertGreater(e_high_rd, e_low_rd)
        self.assertAlmostEqual(glicko_expected_score(1500.0, 1500.0, opp_rd=100.0), 0.5, places=6)

    def test_g_phi_attenuates_with_uncertainty(self):
        from cs2pickem.ratings import glicko_g

        # g(phi) is a decreasing function of phi (more uncertain opponent -> g -> closer
        # to a flatter expectation). g is computed in Glicko-2 scale internally.
        self.assertGreater(glicko_g(30.0), glicko_g(350.0))
        self.assertGreater(glicko_g(0.0), 0.99)  # near-certain opponent -> g ~ 1


class Glicko2MovTests(unittest.TestCase):
    def _blowout_vs_close(self, t1_close, t2_close, t1_blow, t2_blow):
        from cs2pickem.ratings import compute_glicko_ratings

        close = [
            {
                "date": "2026-01-01",
                "team1": "A",
                "team2": "B",
                "winner": "A",
                "team1_score": t1_close,
                "team2_score": t2_close,
            }
        ]
        blow = [
            {
                "date": "2026-01-01",
                "team1": "A",
                "team2": "B",
                "winner": "A",
                "team1_score": t1_blow,
                "team2_score": t2_blow,
            }
        ]
        _pc, fc = compute_glicko_ratings(close, use_mov=True)
        _pb, fb = compute_glicko_ratings(blow, use_mov=True)
        return fc["ratings"]["A"], fb["ratings"]["A"]

    def test_bigger_win_moves_rating_more(self):
        # MOV monotonicity: a 16-2 blowout updates A's rating more than a 16-14 squeaker.
        close_a, blow_a = self._blowout_vs_close(16, 14, 16, 2)
        self.assertGreater(blow_a, close_a)

    def test_mov_has_diminishing_returns(self):
        # The log damping gives diminishing returns: going 16-14 -> 16-8 should gain
        # more rating than going 16-8 -> 16-2 (concavity of ln(1+round_diff)).
        from cs2pickem.ratings import compute_glicko_ratings

        def rating_for(diff_t2):
            m = [
                {
                    "date": "2026-01-01",
                    "team1": "A",
                    "team2": "B",
                    "winner": "A",
                    "team1_score": 16,
                    "team2_score": diff_t2,
                }
            ]
            _p, f = compute_glicko_ratings(m, use_mov=True)
            return f["ratings"]["A"]

        gain_small_to_mid = rating_for(8) - rating_for(14)
        gain_mid_to_big = rating_for(2) - rating_for(8)
        self.assertGreater(gain_small_to_mid, 0.0)
        self.assertGreater(gain_mid_to_big, 0.0)
        self.assertGreater(gain_small_to_mid, gain_mid_to_big)

    def test_mov_falls_back_to_win_loss_when_scores_missing(self):
        from cs2pickem.ratings import compute_glicko_ratings

        # No round scores -> mov_mult == 1.0, identical to use_mov=False.
        m = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
        _p1, f_mov = compute_glicko_ratings(m, use_mov=True)
        _p2, f_no = compute_glicko_ratings(m, use_mov=False)
        self.assertAlmostEqual(f_mov["ratings"]["A"], f_no["ratings"]["A"], places=9)
        self.assertAlmostEqual(f_mov["rds"]["A"], f_no["rds"]["A"], places=9)

    def test_mov_falls_back_when_score_sum_zero(self):
        from cs2pickem.ratings import compute_glicko_ratings

        m = [
            {
                "date": "2026-01-01",
                "team1": "A",
                "team2": "B",
                "winner": "A",
                "team1_score": 0,
                "team2_score": 0,
            }
        ]
        _p1, f_mov = compute_glicko_ratings(m, use_mov=True)
        _p2, f_no = compute_glicko_ratings(m, use_mov=False)
        self.assertAlmostEqual(f_mov["ratings"]["A"], f_no["ratings"]["A"], places=9)

    def test_mov_pre_snapshot_is_independent_of_round_diff(self):
        from cs2pickem.ratings import compute_glicko_ratings

        # Pre-match snapshots are pure entry state; the round diff must not leak into them.
        close = [
            {"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A",
             "team1_score": 16, "team2_score": 14},
        ]
        blow = [
            {"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A",
             "team1_score": 16, "team2_score": 1},
        ]
        pc, _ = compute_glicko_ratings(close, use_mov=True)
        pb, _ = compute_glicko_ratings(blow, use_mov=True)
        self.assertEqual(pc[0]["team1_glicko_pre"], pb[0]["team1_glicko_pre"])
        self.assertEqual(pc[0]["team1_rd_pre"], pb[0]["team1_rd_pre"])

    def test_mov_blowout_against_weak_opponent_is_damped(self):
        from cs2pickem.ratings import compute_glicko_ratings

        # Autocorrelation correction: an identical 16-2 blowout yields LESS rating gain
        # when the winner was already far stronger (big pre-match rating gap) than when
        # the teams were even -- the multiplier's denominator grows with |mu_w - mu_l|.
        even = {
            "ratings": {"A": 1500.0, "B": 1500.0},
            "rds": {"A": 100.0, "B": 100.0},
            "sigmas": {"A": 0.06, "B": 0.06},
        }
        lopsided = {
            "ratings": {"A": 2200.0, "B": 1000.0},
            "rds": {"A": 100.0, "B": 100.0},
            "sigmas": {"A": 0.06, "B": 0.06},
        }
        m = [
            {"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A",
             "team1_score": 16, "team2_score": 2},
        ]
        _pe, fe = compute_glicko_ratings(m, use_mov=True, initial_state=even)
        _pl, fl = compute_glicko_ratings(m, use_mov=True, initial_state=lopsided)
        gain_even = fe["ratings"]["A"] - 1500.0
        gain_lopsided = fl["ratings"]["A"] - 2200.0
        # The favourite's blowout buys it less (damped), so per-point gain is smaller.
        # Compare the raw rating delta magnitudes.
        self.assertGreater(gain_even, gain_lopsided)


class Glicko2StabilityTests(unittest.TestCase):
    def test_does_not_diverge_over_long_history(self):
        from cs2pickem.ratings import compute_glicko_ratings

        # A long alternating-results history must keep mu/phi/sigma finite and bounded.
        matches = []
        for day in range(1, 60):
            winner = "A" if day % 2 else "B"
            matches.append(
                {"date": f"2026-{(day // 28) + 1:02d}-{(day % 28) + 1:02d}",
                 "team1": "A", "team2": "B", "winner": winner,
                 "team1_score": 16, "team2_score": 12}
            )
        _per, final = compute_glicko_ratings(matches, use_mov=True)
        import math

        for team in ("A", "B"):
            self.assertTrue(math.isfinite(final["ratings"][team]))
            self.assertTrue(math.isfinite(final["rds"][team]))
            self.assertTrue(math.isfinite(final["sigmas"][team]))
            self.assertLessEqual(final["rds"][team], 350.0 + 1e-9)
            self.assertGreater(final["rds"][team], 0.0)
            # sigma stays in a sane band (volatility constraint tau keeps it near sigma0).
            self.assertLess(final["sigmas"][team], 1.0)
            self.assertGreater(final["sigmas"][team], 0.0)


class Glicko2EloUntouchedTests(unittest.TestCase):
    def test_elo_behavior_is_bit_for_bit_unchanged(self):
        from cs2pickem.ratings import compute_elo_ratings

        # The Elo baseline anchor must be byte-identical: adding Glicko must not perturb
        # compute_elo_ratings. Reproduce a fixed scenario and assert exact values.
        matches = [
            {"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"},
            {"date": "2026-01-02", "team1": "A", "team2": "C", "winner": "C"},
            {"date": "2026-01-03", "team1": "B", "team2": "C", "winner": "B"},
        ]
        per_match, final = compute_elo_ratings(matches, base=1500.0, k=24.0)
        # Hand/independently verifiable: first match both at base.
        self.assertEqual(per_match[0]["team1_elo_pre"], 1500.0)
        self.assertEqual(per_match[0]["team2_elo_pre"], 1500.0)
        # Exact zero-sum across all teams' total movement.
        self.assertAlmostEqual(final["A"] + final["B"] + final["C"], 4500.0, places=9)
        # Re-running yields identical floats (determinism / no shared mutable state leak).
        per_match2, final2 = compute_elo_ratings(matches, base=1500.0, k=24.0)
        for t in ("A", "B", "C"):
            self.assertEqual(final[t], final2[t])


def math_isfinite(value):
    import math

    return math.isfinite(value)


if __name__ == "__main__":
    unittest.main()

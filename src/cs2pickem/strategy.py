from __future__ import annotations

import math

from typing import Dict, List, Mapping, Optional, Sequence


DEFAULT_SLOTS = {"3-0": 2, "advance": 6, "0-3": 2}

# Pick'em selection objectives.
#   ``expected_hits`` (default) -- target the EXPECTED number of correct slots
#     via per-slot greedy top-N on the (risk-adjusted) marginal score. NOTE on
#     optimality: greedy-by-margin is exactly optimal for the separable
#     expected-hit objective (E[hits] = Σ_slot Σ_team∈slot P(team hits slot))
#     ONLY within a single category. Across categories it is a HEURISTIC, not a
#     proof: ``choose_pickems`` processes categories in a FIXED order
#     (3-0 -> 0-3 -> advance) and dedupes via a shared ``picked`` set, so an
#     earlier category can claim a strong team an unfilled later category would
#     have used more profitably -- the cross-category dedup couples the
#     categories. Additionally the score being ordered is risk-adjusted
#     (upset/stage/player multipliers are documented marginal-probability priors,
#     NOT a joint distribution), so it is not raw P(correct) even per category.
#     This path is the LOCKED historic heuristic (BYTE-FOR-BYTE unchanged,
#     regression baseline, review red line a) -- intentionally frozen, NOT a
#     claim of global optimality.
#   ``threshold_prob`` -- maximise P(correct slots >= K) over the swiss
#     ``joint_samples`` (the full per-bracket outcome vectors). Used for the
#     "make the pass line" scenario. Search = greedy EV seed + deterministic
#     single-pick local swaps on the Monte-Carlo P(hits >= K); it NEVER enumerates
#     the full combinatorial ticket space (red line b).
#   ``leveraged`` -- contrarian / EV-leverage tilt. Same joint-sample +
#     local-swap machinery, but the objective is an expected pool-share-style
#     reward that pays MORE when the ticket hits a low-consensus outcome the
#     "field" tends to miss. With no real crowd data the field is approximated by
#     the marginal hit probabilities (documented degradation).
PICKEM_OBJECTIVES = ("expected_hits", "threshold_prob", "leveraged")
DEFAULT_PICKEM_OBJECTIVE = "expected_hits"

# Category -> per-team boolean key inside a joint-sample outcome vector. The swiss
# encoder (swiss._encode_joint_outcome) stores exactly these keys.
_PICKEM_CATEGORY_KEYS = ("3-0", "advance", "0-3")

# Fusion methods for blending the model probability toward the market signal.
#   ``legacy_clip`` -- the historic arithmetic +/-``max_adjustment`` nudge (default,
#     bit-for-bit unchanged behaviour).
#   ``logit_pool``  -- a logarithmic opinion pool (geometric mean of the two
#     probabilistic experts): logit(p_fused) = w*logit(p_model) + (1-w)*logit(p_market)
#     with ``w`` (the model weight) FROZEN to a literature prior, never fitted.
FUSION_METHODS = ("legacy_clip", "logit_pool")
DEFAULT_FUSION_METHOD = "legacy_clip"
# Frozen literature prior on the MODEL's weight in the logit pool. ~0.35 leans on
# the market (~0.65 weight on the de-vigged fair probability) because the market is
# a strong, near-calibrated aggregator. This is a documented hyperparameter default
# -- it is deliberately NOT estimated on a single event's tiny holdout (review point
# 8: that fits noise). Multi-season re-litigation is deferred to WF-2F.
DEFAULT_MODEL_WEIGHT = 0.35

# --------------------------------------------------------------------------- #
# Production fusion defaults (WF-2F verdict).
#
# The bare-call library defaults above (DEFAULT_FUSION_METHOD='legacy_clip',
# DEFAULT_MODEL_WEIGHT=0.35) are LOCKED by behaviour-contract tests and the
# tuning diagnostic口径 -- they are intentionally NOT changed here. Instead the
# two PRODUCTION fusion call points (forecast.forecast_fixtures and
# pickem.model_driven_pickems / swiss_predictor) opt into the logit pool by
# passing these constants explicitly. WF-2F adjudicated the logarithmic opinion
# pool a significant improvement over the legacy arithmetic clip, with a model
# weight of ~0.30 (i.e. leaning further on the market than the 0.35 frozen
# diagnostic prior). Centralising them here keeps the production fusion口径 in one
# referenceable place while the library/tuning defaults stay byte-identical.
PRODUCTION_FUSION_METHOD = "logit_pool"
PRODUCTION_MODEL_WEIGHT = 0.30
# Logit-pool guard: clip probabilities into [_LOGIT_POOL_EPS, 1 - _LOGIT_POOL_EPS]
# before taking logits and clip the pooled output so 0/1 inputs never overflow.
_LOGIT_POOL_EPS = 1e-6


def adjust_probability_with_market(
    model_probability: float,
    odds_team1: float,
    odds_team2: float,
    max_adjustment: float = 0.03,
    fusion_method: str = DEFAULT_FUSION_METHOD,
    model_weight: float = DEFAULT_MODEL_WEIGHT,
) -> float:
    market_probability = _market_probability(odds_team1, odds_team2)
    return adjust_probability_toward_market_probability(
        model_probability,
        market_probability,
        max_adjustment=max_adjustment,
        fusion_method=fusion_method,
        model_weight=model_weight,
    )


def adjust_probability_toward_market_probability(
    model_probability: float,
    market_probability: float,
    max_adjustment: float = 0.03,
    fusion_method: str = DEFAULT_FUSION_METHOD,
    model_weight: float = DEFAULT_MODEL_WEIGHT,
) -> float:
    """Blend ``model_probability`` toward ``market_probability``.

    ``fusion_method``:

    * ``legacy_clip`` (default) -- the historic arithmetic nudge: move at most
      ``max_adjustment`` toward the market, then clip to [0, 1]. Behaviour is
      bit-for-bit unchanged.
    * ``logit_pool`` -- a logarithmic opinion pool (geometric-mean consensus of
      two probabilistic experts): ``logit(p_fused) = w*logit(p_model) +
      (1 - w)*logit(p_market)`` with the model weight ``w = model_weight``
      FROZEN to a literature prior (default ~0.35, i.e. pro-market). ``w`` is a
      documented hyperparameter, NOT fitted on this event (review red-line). The
      ``market_probability`` should already be the de-vigged fair probability
      (see :func:`cs2pickem.odds.devig_market`). ``max_adjustment`` is ignored on
      this path.
    """
    method = _resolve_fusion_method(fusion_method)
    if method == "logit_pool":
        return _logit_pool(model_probability, market_probability, model_weight)
    delta = market_probability - model_probability
    capped_delta = max(-max_adjustment, min(max_adjustment, delta))
    return _clip(model_probability + capped_delta)


def _resolve_fusion_method(fusion_method: Optional[str]) -> str:
    resolved = str(fusion_method or DEFAULT_FUSION_METHOD).strip().lower()
    if resolved not in FUSION_METHODS:
        raise ValueError(f"unknown fusion method: {fusion_method!r}; expected one of {FUSION_METHODS}")
    return resolved


def _logit_pool(model_probability: float, market_probability: float, model_weight: float) -> float:
    """Logarithmic opinion pool of two probabilities with a frozen model weight.

    Clamps ``model_weight`` to [0, 1] (boundary check; not an MLE), clips both
    inputs into ``[_LOGIT_POOL_EPS, 1 - _LOGIT_POOL_EPS]`` before taking logits,
    and clips the pooled output to avoid 0/1 overflow. With ``w = 1`` the result
    collapses to the (clipped) model probability and with ``w = 0`` to the
    (clipped) market probability.
    """
    weight = max(0.0, min(1.0, float(model_weight)))
    pooled_logit = weight * _logit(model_probability) + (1.0 - weight) * _logit(market_probability)
    return _clip(_sigmoid(pooled_logit))


def _logit(probability: float) -> float:
    clipped = min(1.0 - _LOGIT_POOL_EPS, max(_LOGIT_POOL_EPS, float(probability)))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value < -60.0:
        return 0.0
    if value > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def single_match_pick(
    probability_team1: float,
    team1: str,
    team2: str,
    threshold: float = 0.52,
    minimum_margin: float | None = None,
    player_form_score_diff: float | None = None,
    player_form_sample_confidence: float | None = None,
    player_form_counter_min_confidence: float = 0.0,
    avoid_player_form_counter_signal: bool = False,
    avoid_player_status_risk: bool = False,
    player_status_min_confidence: float = 0.4,
    player_status_min_margin: float = 0.06,
    team1_player_sample_confidence: float | None = None,
    team2_player_sample_confidence: float | None = None,
    team1_substitute_flag: float | None = None,
    team2_substitute_flag: float | None = None,
) -> str:
    effective_threshold = 0.5 + max(0.0, minimum_margin) if minimum_margin is not None else threshold
    if max(probability_team1, 1.0 - probability_team1) <= effective_threshold:
        return "avoid"
    if avoid_player_form_counter_signal and player_form_score_diff is not None:
        sample_confidence = 1.0 if player_form_sample_confidence is None else _num(player_form_sample_confidence, 0.0)
        directional_form_score = player_form_score_diff if probability_team1 >= 0.5 else -player_form_score_diff
        if sample_confidence >= player_form_counter_min_confidence and directional_form_score < 0:
            return "avoid"
    if avoid_player_status_risk and _picked_player_status_risk(
        probability_team1=probability_team1,
        team1_player_sample_confidence=team1_player_sample_confidence,
        team2_player_sample_confidence=team2_player_sample_confidence,
        team1_substitute_flag=team1_substitute_flag,
        team2_substitute_flag=team2_substitute_flag,
        min_confidence=player_status_min_confidence,
    ):
        if abs(probability_team1 - 0.5) <= player_status_min_margin:
            return "avoid"
    return team1 if probability_team1 >= 0.5 else team2


def choose_pickems(
    team_probabilities: Mapping[str, Mapping[str, float]],
    rankings: Optional[Mapping[str, int]] = None,
    slots: Optional[Mapping[str, int]] = None,
    upset_rank_limit: int = 15,
    stage: str = "default",
    team_features: Optional[Mapping[str, Mapping[str, float]]] = None,
    objective: str = DEFAULT_PICKEM_OBJECTIVE,
    joint_samples: Optional[Sequence[Mapping[str, Mapping[str, object]]]] = None,
    threshold: Optional[int] = None,
    crowd_probabilities: Optional[Mapping[str, Mapping[str, float]]] = None,
    leverage_strength: float = 1.0,
    max_swaps: int = 64,
) -> Dict[str, List[str]]:
    """Select a pick'em ticket under one of :data:`PICKEM_OBJECTIVES`.

    ``objective`` (default ``expected_hits``):

    * ``expected_hits`` -- per-slot greedy top-N on the risk-adjusted marginal
      score. This is the historic behaviour and is BYTE-FOR-BYTE unchanged; the
      ``joint_samples``/``threshold``/``crowd_*``/``leverage_*`` arguments are
      ignored on this path. Greedy is exactly optimal WITHIN a category (the
      expected-hit objective is separable there) but is a HEURISTIC across
      categories -- the fixed processing order plus cross-category ``picked``
      dedup couples them, so this is the locked historic heuristic, not a
      global-optimality guarantee (see the module-level PICKEM_OBJECTIVES note).
    * ``threshold_prob`` -- start from the ``expected_hits`` ticket as a seed and
      run deterministic single-pick local swaps to maximise the Monte-Carlo
      ``P(hits >= K)`` estimated over ``joint_samples`` (``K = threshold``,
      default = total slot count, i.e. "all picks correct"). Requires
      ``joint_samples`` (raises ``ValueError`` otherwise). Never enumerates the
      full ticket space.
    * ``leveraged`` -- same seed + local-swap search, but the objective is an
      expected pool-share reward that rewards hitting outcomes the field misses
      (contrarian). Requires ``joint_samples``.

    The return shape is always ``{"3-0": [...], "advance": [...], "0-3": [...]}``.
    """
    resolved_objective = _resolve_pickem_objective(objective)
    slots = dict(slots or DEFAULT_SLOTS)
    rankings = dict(rankings or {})
    team_features = dict(team_features or {})

    seed_ticket = _choose_pickems_expected_hits(
        team_probabilities,
        rankings=rankings,
        slots=slots,
        upset_rank_limit=upset_rank_limit,
        stage=stage,
        team_features=team_features,
    )
    if resolved_objective == "expected_hits":
        return seed_ticket

    if not joint_samples:
        raise ValueError(
            f"objective={resolved_objective!r} requires non-empty joint_samples "
            "(simulate_swiss(..., collect_joint=True))"
        )
    return _choose_pickems_joint(
        resolved_objective,
        seed_ticket=seed_ticket,
        team_probabilities=team_probabilities,
        rankings=rankings,
        slots=slots,
        upset_rank_limit=upset_rank_limit,
        stage=stage,
        team_features=team_features,
        joint_samples=joint_samples,
        threshold=threshold,
        crowd_probabilities=crowd_probabilities,
        leverage_strength=leverage_strength,
        max_swaps=max_swaps,
    )


def _resolve_pickem_objective(objective: Optional[str]) -> str:
    resolved = str(objective or DEFAULT_PICKEM_OBJECTIVE).strip().lower()
    if resolved not in PICKEM_OBJECTIVES:
        raise ValueError(
            f"unknown pickem objective: {objective!r}; expected one of {PICKEM_OBJECTIVES}"
        )
    return resolved


def _choose_pickems_expected_hits(
    team_probabilities: Mapping[str, Mapping[str, float]],
    rankings: Mapping[str, int],
    slots: Mapping[str, int],
    upset_rank_limit: int,
    stage: str,
    team_features: Mapping[str, Mapping[str, float]],
) -> Dict[str, List[str]]:
    picked: set[str] = set()

    three_zero = _top_teams(team_probabilities, "3-0", slots.get("3-0", 0), rankings, upset_rank_limit, picked, stage=stage, team_features=team_features)
    picked.update(three_zero)

    zero_three = _top_teams(team_probabilities, "0-3", slots.get("0-3", 0), rankings, upset_rank_limit, picked, prefer_weak=True, stage=stage, team_features=team_features)
    picked.update(zero_three)

    advance = _top_teams(team_probabilities, "advance", slots.get("advance", 0), rankings, upset_rank_limit, picked, stage=stage, team_features=team_features)

    return {"3-0": three_zero, "advance": advance, "0-3": zero_three}


def describe_pickems(
    team_probabilities: Mapping[str, Mapping[str, float]],
    pickems: Mapping[str, List[str]],
    rankings: Optional[Mapping[str, int]] = None,
    risk_details: Optional[Mapping[str, List[Mapping[str, object]]]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    rankings = dict(rankings or {})
    details: Dict[str, List[Dict[str, object]]] = {}
    selected_anywhere = {team for teams in pickems.values() for team in teams}
    for category, teams in pickems.items():
        selected = list(teams)
        score_lookup = _risk_score_lookup(risk_details, category)
        if score_lookup:
            unselected_scores = [
                score
                for team, score in score_lookup.items()
                if team not in selected_anywhere
            ]
            next_best = max(unselected_scores) if unselected_scores else None
        else:
            unselected_probabilities = [
                float(values.get(category, 0.0))
                for team, values in team_probabilities.items()
                if team not in selected
            ]
            next_best = max(unselected_probabilities) if unselected_probabilities else None
        details[category] = []
        for team in selected:
            probability = float(team_probabilities.get(team, {}).get(category, 0.0))
            selection_score = score_lookup.get(team) if score_lookup else probability
            details[category].append(
                {
                    "team": team,
                    "category": category,
                    "probability": probability,
                    "rank": rankings.get(team),
                    "next_best_probability": next_best if not score_lookup else None,
                    "selection_score": selection_score,
                    "next_best_score": next_best if score_lookup else None,
                    "selection_margin": selection_score - next_best if next_best is not None and selection_score is not None else None,
                }
            )
    return details


def _risk_score_lookup(risk_details: Optional[Mapping[str, List[Mapping[str, object]]]], category: str) -> Dict[str, float]:
    if not risk_details:
        return {}
    entries = risk_details.get(category, [])
    lookup: Dict[str, float] = {}
    for entry in entries:
        team = entry.get("team")
        if team in (None, ""):
            continue
        try:
            lookup[str(team)] = float(entry.get("final_score"))
        except (TypeError, ValueError):
            continue
    return lookup


def describe_pickem_risk(
    team_probabilities: Mapping[str, Mapping[str, float]],
    rankings: Optional[Mapping[str, int]] = None,
    upset_rank_limit: int = 15,
    stage: str = "default",
    team_features: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    rankings = dict(rankings or {})
    team_features = dict(team_features or {})
    best_rank = _best_rank(team_probabilities, rankings)
    details: Dict[str, List[Dict[str, object]]] = {}
    for key in ("3-0", "advance", "0-3"):
        entries = []
        for team, values in team_probabilities.items():
            rank = rankings.get(team, 80)
            base_probability = float(values.get(key, 0.0))
            features = team_features.get(team, {})
            stage_adjustment = _stage_adjustment(stage, key, features, rank)
            player_form_adjustment = _player_form_adjustment(key, features)
            player_status = _player_status_details(features)
            upset_rank_gap = max(0, rank - best_rank)
            upset_penalty_multiplier = _upset_penalty_multiplier(key, upset_rank_gap, upset_rank_limit)
            prefer_weak_multiplier = _prefer_weak_multiplier(rank) if key == "0-3" else 1.0
            player_availability_multiplier = _player_availability_multiplier(key, features)
            final_score = (
                (base_probability + stage_adjustment + player_form_adjustment)
                * upset_penalty_multiplier
                * prefer_weak_multiplier
                * player_availability_multiplier
            )
            entries.append(
                {
                    "team": team,
                    "category": key,
                    "rank": rank,
                    "base_probability": base_probability,
                    "stage_adjustment": stage_adjustment,
                    "player_form_adjustment": player_form_adjustment,
                    "player_form_score": player_status["player_form_score"],
                    "player_form_trend": player_status["player_form_trend"],
                    "player_sample_confidence": player_status["player_sample_confidence"],
                    "substitute_flag": player_status["substitute_flag"],
                    "player_status_risk": player_status["player_status_risk"],
                    "upset_rank_gap": upset_rank_gap,
                    "upset_penalty_multiplier": upset_penalty_multiplier,
                    "prefer_weak_multiplier": prefer_weak_multiplier,
                    "player_availability_multiplier": player_availability_multiplier,
                    "final_score": final_score,
                }
            )
        details[key] = sorted(entries, key=lambda entry: (-float(entry["final_score"]), int(entry["rank"]), str(entry["team"])))
    return details


# --------------------------------------------------------------------------- #
# Joint-sample pick'em objectives (threshold_prob / leveraged).
#
# A pick'em "ticket" is a category -> [team, ...] mapping; a single pick (cat,
# team) "hits" a finished bracket sample iff sample[team][cat] is truthy. The
# swiss MC encoder (swiss._encode_joint_outcome) stores those per-team booleans
# for "3-0" / "advance" / "0-3". All objectives below evaluate the ticket against
# the SHARED set of joint_samples instead of enumerating the ~10^7 combinatorial
# ticket space (review red line b). Search is a greedy expected-hits seed plus
# deterministic single-pick local swaps -- O(samples x swaps), not combinatorial.
#
# MONTE-CARLO CAVEAT: every probability below is a sample-mean estimate. Its
# standard error is ~sqrt(p(1-p)/N); evaluate_ticket_distribution reports N and a
# 95% normal-approx CI so callers can judge whether two tickets are separable.
# --------------------------------------------------------------------------- #


def _flatten_ticket(ticket: Mapping[str, Sequence[str]]) -> List[tuple[str, str]]:
    """Flatten a ticket into an ordered list of (category, team) picks."""
    picks: List[tuple[str, str]] = []
    for category in _PICKEM_CATEGORY_KEYS:
        for team in ticket.get(category, []) or []:
            picks.append((category, str(team)))
    return picks


def ticket_hits_in_sample(
    ticket: Mapping[str, Sequence[str]],
    sample: Mapping[str, Mapping[str, object]],
) -> int:
    """Number of (category, team) picks in ``ticket`` correct in one bracket sample."""
    hits = 0
    for category, team in _flatten_ticket(ticket):
        outcome = sample.get(team)
        if outcome is not None and bool(outcome.get(category)):
            hits += 1
    return hits


def ticket_threshold_probability(
    ticket: Mapping[str, Sequence[str]],
    joint_samples: Sequence[Mapping[str, Mapping[str, object]]],
    threshold: int,
) -> float:
    """Monte-Carlo estimate of P(ticket correct picks >= ``threshold``)."""
    if not joint_samples:
        return 0.0
    successes = sum(1 for sample in joint_samples if ticket_hits_in_sample(ticket, sample) >= threshold)
    return successes / len(joint_samples)


def evaluate_ticket_distribution(
    ticket: Mapping[str, Sequence[str]],
    joint_samples: Sequence[Mapping[str, Mapping[str, object]]],
    threshold: Optional[int] = None,
) -> Dict[str, object]:
    """Summarise a ticket's hit distribution over the shared joint samples.

    Reports the expected hits, the full hit histogram, and -- for ``threshold``
    (default = total picks) -- the Monte-Carlo P(hits >= K) with its sample size
    and a 95% normal-approximation confidence interval (the estimate is a
    sample mean; two tickets whose CIs overlap are not statistically separable).
    """
    picks = _flatten_ticket(ticket)
    total_picks = len(picks)
    k = total_picks if threshold is None else int(threshold)
    n = len(joint_samples)
    histogram: Dict[int, int] = {}
    total_hits = 0
    successes = 0
    for sample in joint_samples:
        hits = ticket_hits_in_sample(ticket, sample)
        histogram[hits] = histogram.get(hits, 0) + 1
        total_hits += hits
        if hits >= k:
            successes += 1
    probability = successes / n if n else 0.0
    half_width = 1.96 * math.sqrt(probability * (1.0 - probability) / n) if n else 0.0
    return {
        "total_picks": total_picks,
        "threshold": k,
        "samples": n,
        "expected_hits": (total_hits / n) if n else 0.0,
        "hit_histogram": dict(sorted(histogram.items())),
        "threshold_probability": probability,
        "threshold_probability_ci95": (
            max(0.0, probability - half_width),
            min(1.0, probability + half_width),
        ),
    }


def _candidate_pool(
    team_probabilities: Mapping[str, Mapping[str, float]],
    rankings: Mapping[str, int],
    upset_rank_limit: int,
    stage: str,
    team_features: Mapping[str, Mapping[str, float]],
) -> Dict[str, List[str]]:
    """Per-category candidate teams ordered by risk-adjusted marginal score.

    Reuses the exact ``expected_hits`` scoring so the local search explores the
    same risk-aware ordering the EV seed came from (the contrarian tilt for the
    leveraged objective is applied later, in the swap acceptance test).
    """
    pool: Dict[str, List[str]] = {}
    for category in _PICKEM_CATEGORY_KEYS:
        prefer_weak = category == "0-3"
        ordered = _top_teams(
            team_probabilities,
            category,
            len(team_probabilities),
            rankings,
            upset_rank_limit,
            excluded=set(),
            prefer_weak=prefer_weak,
            stage=stage,
            team_features=team_features,
        )
        pool[category] = ordered
    return pool


def _ticket_picked_teams(ticket: Mapping[str, Sequence[str]]) -> set[str]:
    return {str(team) for teams in ticket.values() for team in (teams or [])}


def _local_swap_search(
    seed_ticket: Mapping[str, List[str]],
    candidate_pool: Mapping[str, List[str]],
    score_fn,
    max_swaps: int,
) -> Dict[str, List[str]]:
    """Deterministic single-pick local search maximising ``score_fn(ticket)``.

    Starting from ``seed_ticket`` (the EV greedy), repeatedly try replacing one
    picked team in one category with the best-scoring non-picked candidate. Keep
    the first strictly-improving swap; stop at a local optimum or ``max_swaps``.
    Cross-category duplicate teams are never introduced (keeps the dedup
    invariant ``choose_pickems`` guarantees). This is O(samples x swaps), never a
    full ticket enumeration (review red line b).
    """
    ticket: Dict[str, List[str]] = {cat: list(teams) for cat, teams in seed_ticket.items()}
    best_score = score_fn(ticket)
    for _ in range(max(0, max_swaps)):
        improved = False
        for category in _PICKEM_CATEGORY_KEYS:
            slot_teams = ticket.get(category, [])
            for position in range(len(slot_teams)):
                current_team = slot_teams[position]
                others = _ticket_picked_teams(ticket) - {current_team}
                for candidate in candidate_pool.get(category, []):
                    if candidate == current_team or candidate in others:
                        continue
                    trial = {cat: list(teams) for cat, teams in ticket.items()}
                    trial[category][position] = candidate
                    trial_score = score_fn(trial)
                    if trial_score > best_score + 1e-12:
                        ticket = trial
                        best_score = trial_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return {category: list(ticket.get(category, [])) for category in _PICKEM_CATEGORY_KEYS}


def _field_hit_probabilities(
    crowd_probabilities: Optional[Mapping[str, Mapping[str, float]]],
    joint_samples: Sequence[Mapping[str, Mapping[str, object]]],
) -> Mapping[str, Mapping[str, float]]:
    """Per-(team, category) probability the *field* (crowd) gets that pick right.

    With real crowd pick-popularity data this is supplied via
    ``crowd_probabilities``. Absent that, the field is approximated by the
    model's own marginal hit rate over the joint samples (a documented
    degradation: it assumes the crowd picks the chalk, so "contrarian" reduces to
    "fade the marginal favourite").
    """
    if crowd_probabilities:
        return crowd_probabilities
    counts: Dict[str, Dict[str, int]] = {}
    n = len(joint_samples)
    for sample in joint_samples:
        for team, outcome in sample.items():
            bucket = counts.setdefault(team, {key: 0 for key in _PICKEM_CATEGORY_KEYS})
            for key in _PICKEM_CATEGORY_KEYS:
                if bool(outcome.get(key)):
                    bucket[key] += 1
    return {
        team: {key: (value / n if n else 0.0) for key, value in bucket.items()}
        for team, bucket in counts.items()
    }


def _leveraged_reward(
    ticket: Mapping[str, Sequence[str]],
    joint_samples: Sequence[Mapping[str, Mapping[str, object]]],
    field_probabilities: Mapping[str, Mapping[str, float]],
    leverage_strength: float,
) -> float:
    """Expected pool-share-style reward for a contrarian ticket.

    For each correct pick in a sample we award ``1 / field_prob`` raised to
    ``leverage_strength`` -- i.e. nailing a pick the field rarely gets right pays
    far more than agreeing with the chalk. Averaging over the shared joint
    samples yields the expected leveraged reward (a pool-share proxy: my upside
    grows as the field's hit probability shrinks). With ``leverage_strength=0``
    this collapses to plain expected hits.

    ROBUSTNESS / SAMPLING FLOOR: when the field probability is itself a
    Monte-Carlo estimate over ``N`` joint samples, the finest non-zero rate it
    can resolve is ``1 / N``. Without a floor, a longshot the field "hits" only
    1-2 times out of N gets a weight that explodes faster than its own hit rate
    shrinks, so for ``leverage_strength > 1`` the objective would chase whichever
    rare outcomes happened to land in the finite sample (pure MC noise). We
    therefore floor ``field_prob`` at ``1 / N`` before exponentiating. Even so,
    the reward grows steeply in ``leverage_strength``; the recommended range is
    roughly ``[0, 1.5]`` -- much larger values degrade toward "always fade the
    rarest outcome" regardless of true edge.
    """
    if not joint_samples:
        return 0.0
    picks = _flatten_ticket(ticket)
    strength = max(0.0, float(leverage_strength))
    # Floor the field probability at the finest MC-resolvable non-zero rate so a
    # 1-of-N longshot cannot dominate via sampling noise (see docstring).
    field_floor = 1.0 / len(joint_samples)
    weights: Dict[tuple[str, str], float] = {}
    for category, team in picks:
        field_prob = float(field_probabilities.get(team, {}).get(category, 0.5))
        field_prob = max(field_prob, field_floor)
        weights[(category, team)] = (1.0 / field_prob) ** strength
    total = 0.0
    for sample in joint_samples:
        for category, team in picks:
            outcome = sample.get(team)
            if outcome is not None and bool(outcome.get(category)):
                total += weights[(category, team)]
    return total / len(joint_samples)


def _choose_pickems_joint(
    objective: str,
    seed_ticket: Mapping[str, List[str]],
    team_probabilities: Mapping[str, Mapping[str, float]],
    rankings: Mapping[str, int],
    slots: Mapping[str, int],
    upset_rank_limit: int,
    stage: str,
    team_features: Mapping[str, Mapping[str, float]],
    joint_samples: Sequence[Mapping[str, Mapping[str, object]]],
    threshold: Optional[int],
    crowd_probabilities: Optional[Mapping[str, Mapping[str, float]]],
    leverage_strength: float,
    max_swaps: int,
) -> Dict[str, List[str]]:
    candidate_pool = _candidate_pool(
        team_probabilities,
        rankings=rankings,
        upset_rank_limit=upset_rank_limit,
        stage=stage,
        team_features=team_features,
    )
    if objective == "threshold_prob":
        # Default K = the seed ticket's actual flattened pick count, NOT
        # sum(slots). When a category cannot fill its slots (fewer candidates
        # than slots) those counts diverge, and optimising P(hits >= sum(slots))
        # would target an unreachable threshold (identically 0 -> flat objective)
        # while evaluate_ticket_distribution reports P(hits >= flattened picks).
        # Defaulting both to the flattened pick count keeps the optimiser's K and
        # the reported distribution's K in agreement. In the canonical 16-team run
        # the ticket fills every slot, so this equals sum(slots) (no behaviour
        # change there).
        total_picks = len(_flatten_ticket(seed_ticket))
        k = total_picks if threshold is None else int(threshold)

        def score_fn(ticket: Mapping[str, Sequence[str]]) -> float:
            return ticket_threshold_probability(ticket, joint_samples, k)

    else:  # leveraged
        field_probabilities = _field_hit_probabilities(crowd_probabilities, joint_samples)

        def score_fn(ticket: Mapping[str, Sequence[str]]) -> float:
            return _leveraged_reward(ticket, joint_samples, field_probabilities, leverage_strength)

    return _local_swap_search(seed_ticket, candidate_pool, score_fn, max_swaps)


def _top_teams(
    probabilities: Mapping[str, Mapping[str, float]],
    key: str,
    count: int,
    rankings: Mapping[str, int],
    upset_rank_limit: int,
    excluded: set[str],
    prefer_weak: bool = False,
    stage: str = "default",
    team_features: Mapping[str, Mapping[str, float]] | None = None,
) -> List[str]:
    team_features = team_features or {}
    scored = []
    best_rank = _best_rank(probabilities, rankings)
    for team, values in probabilities.items():
        if team in excluded:
            continue
        rank = rankings.get(team, 80)
        score = _candidate_score(
            float(values.get(key, 0.0)),
            key,
            rank,
            best_rank,
            upset_rank_limit,
            stage,
            team_features.get(team, {}),
            prefer_weak=prefer_weak,
        )
        scored.append((score, -rank if prefer_weak else rank, team))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [team for _, _, team in scored[:count]]


def _candidate_score(
    base_probability: float,
    key: str,
    rank: int,
    best_rank: int,
    upset_rank_limit: int,
    stage: str,
    features: Mapping[str, float],
    prefer_weak: bool = False,
) -> float:
    score = base_probability + _stage_adjustment(stage, key, features, rank) + _player_form_adjustment(key, features)
    score *= _upset_penalty_multiplier(key, max(0, rank - best_rank), upset_rank_limit)
    if prefer_weak:
        score *= _prefer_weak_multiplier(rank)
    score *= _player_availability_multiplier(key, features)
    return score


def _best_rank(probabilities: Mapping[str, Mapping[str, float]], rankings: Mapping[str, int]) -> int:
    return min((rankings.get(team, 80) for team in probabilities), default=1)


def _upset_penalty_multiplier(key: str, rank_gap: int, upset_rank_limit: int) -> float:
    if key in {"3-0", "advance"} and rank_gap > upset_rank_limit:
        return 0.75
    return 1.0


def _prefer_weak_multiplier(rank: int) -> float:
    return 1.0 + min(rank, 80) / 200.0


def _market_probability(odds_team1: float, odds_team2: float) -> float:
    """De-vigged fair probability for team1, via the single de-vig source.

    Delegates to :func:`cs2pickem.odds.devig_market` (the canonical de-vig
    implementation) so there is exactly ONE de-vig truth: if the default de-vig
    method ever changes (e.g. to power/shin), the market signal consumed here and
    by the rest of the pipeline stays consistent instead of silently diverging
    from a second hard-coded proportional-normalisation copy. The historic
    ``odds <= 0 -> 0.5`` guard is preserved for malformed inputs.
    """
    if not (odds_team1 > 0 and odds_team2 > 0):
        return 0.5
    from .odds import devig_market

    return float(devig_market(odds_team1, odds_team2)["fair_prob_team1"])


def _picked_player_status_risk(
    probability_team1: float,
    team1_player_sample_confidence: float | None,
    team2_player_sample_confidence: float | None,
    team1_substitute_flag: float | None,
    team2_substitute_flag: float | None,
    min_confidence: float,
) -> bool:
    if probability_team1 >= 0.5:
        sample_confidence = 1.0 if team1_player_sample_confidence is None else _clip(_num(team1_player_sample_confidence, 0.0))
        substitute_flag = _num(team1_substitute_flag, 0.0)
    else:
        sample_confidence = 1.0 if team2_player_sample_confidence is None else _clip(_num(team2_player_sample_confidence, 0.0))
        substitute_flag = _num(team2_substitute_flag, 0.0)
    return sample_confidence < min_confidence or substitute_flag >= 1.0


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _stage_adjustment(stage: str, key: str, features: Mapping[str, float], rank: int) -> float:
    if key == "0-3":
        return 0.0
    normalized_stage = stage.strip().lower()
    if normalized_stage in {"challengers", "challenger", "opening"}:
        bo1 = _num(features.get("bo1_winrate_6m", features.get("bo1_winrate", 0.5)), 0.5)
        map_depth = _num(features.get("map_depth", features.get("map_pool_score", 0.5)), 0.5)
        return max(-0.04, min(0.06, (bo1 - 0.5) * 0.10 + (map_depth - 0.5) * 0.08))
    if normalized_stage in {"legends", "legend", "elimination"}:
        rating = _num(features.get("rating", features.get("team_rating", 1.0)), 1.0)
        rank_bonus = (80.0 - min(max(rank, 1), 80)) / 80.0
        return max(-0.03, min(0.06, (rating - 1.0) * 0.10 + rank_bonus * 0.04))
    return 0.0


def _player_form_adjustment(key: str, features: Mapping[str, float]) -> float:
    form_score = _num(features.get("player_form_score", features.get("form_score", 0.0)), 0.0)
    form_trend = _num(features.get("player_form_trend", features.get("form_trend", 0.0)), 0.0)
    adjustment = max(-0.035, min(0.035, form_score * 0.12 + form_trend * 0.08))
    return -adjustment if key == "0-3" else adjustment


def _player_availability_multiplier(key: str, features: Mapping[str, float]) -> float:
    sample_confidence = _clip(_num(features.get("player_sample_confidence", features.get("sample_confidence", 1.0)), 1.0))
    substitute_flag = 1.0 if _num(features.get("substitute_flag", features.get("player_substitute_flag", 0.0)), 0.0) >= 1.0 else 0.0
    if key == "0-3":
        return min(1.08, 1.0 + (1.0 - sample_confidence) * 0.04 + substitute_flag * 0.03)
    if key == "3-0":
        return max(0.84, 1.0 - (1.0 - sample_confidence) * 0.12 - substitute_flag * 0.08)
    return max(0.90, 1.0 - (1.0 - sample_confidence) * 0.05 - substitute_flag * 0.04)


def _player_status_details(features: Mapping[str, float]) -> Dict[str, object]:
    sample_confidence = _clip(_num(features.get("player_sample_confidence", features.get("sample_confidence", 1.0)), 1.0))
    substitute_flag = 1 if _num(features.get("substitute_flag", features.get("player_substitute_flag", 0.0)), 0.0) >= 1.0 else 0
    return {
        "player_form_score": _num(features.get("player_form_score", features.get("form_score", 0.0)), 0.0),
        "player_form_trend": _num(features.get("player_form_trend", features.get("form_trend", 0.0)), 0.0),
        "player_sample_confidence": sample_confidence,
        "substitute_flag": substitute_flag,
        "player_status_risk": sample_confidence < 0.4 or substitute_flag >= 1,
    }


def _num(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

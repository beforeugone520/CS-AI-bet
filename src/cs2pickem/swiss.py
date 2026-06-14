from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Set, Tuple


@dataclass(frozen=True)
class TeamSeed:
    name: str
    seed: int


@dataclass
class TeamState:
    team: TeamSeed
    wins: int = 0
    losses: int = 0
    opponents: Set[str] = None

    def __post_init__(self) -> None:
        if self.opponents is None:
            self.opponents = set()

    @property
    def active(self) -> bool:
        return self.wins < 3 and self.losses < 3


@dataclass
class SwissSimulationResult:
    team_probabilities: Dict[str, Dict[str, float]]
    simulations: int
    # Optional per-simulation joint outcome vectors (one dict per MC iteration),
    # populated only when simulate_swiss(..., collect_joint=True). Each entry maps
    # team name -> {"record": "W-L", "advance": bool, "eliminate": bool,
    # "3-0": bool, "3-1": bool, "3-2": bool, "0-3": bool, "1-3": bool, "2-3": bool}.
    # Downstream pickem joint optimisation estimates P(hits >= K) over these shared
    # samples instead of enumerating the full combinatorial space.
    joint_samples: List[Dict[str, Dict[str, object]]] = field(default_factory=list)


Predictor = Callable[[TeamSeed, TeamSeed, int, Dict[str, TeamState]], float]

_RECORD_KEYS = ("3-0", "3-1", "3-2", "0-3", "1-3", "2-3")


def simulate_swiss(
    teams: Iterable[TeamSeed],
    predictor: Predictor,
    simulations: int = 100000,
    seed: int = 13,
    pairing: str = "legacy",
    collect_joint: bool = False,
) -> SwissSimulationResult:
    """Monte-Carlo a Swiss stage and return per-team outcome distributions.

    pairing:
      - "legacy" (default): the historical seed-split opening (1v16/2v15…) plus
        seed-ordered score-bucket snake pairing. Behaviour is byte-for-byte
        unchanged from prior releases for backward compatibility.
      - "buchholz": Valve Major rules — 1v9 offset opening (theoretical seeded
        form; see _opening_pairings_1v9), (wins,losses) buckets re-ranked by
        Buchholz difficulty (Σ opp wins − Σ opp losses, seed tiebreak),
        highest-vs-lowest within a bucket, and a best-effort no-rematch
        constraint solved by deterministic backtracking. The no-rematch
        constraint is satisfied whenever a rematch-free perfect matching exists
        (always the case in a standard 16-team Major — empirically 0 rematches);
        on a degenerate bucket with no valid rematch-free matching it falls back
        to greedy snake pairing, which may emit a rematch (documented fallback,
        parity with the front-end pairBucket).

    collect_joint: when True, every simulated bracket's full outcome vector is
      appended to SwissSimulationResult.joint_samples for downstream joint
      pickem optimisation (P(hits >= K) over a shared sample set).
    """
    team_list = sorted(list(teams), key=lambda team: team.seed)
    if len(team_list) % 2 != 0:
        raise ValueError("Swiss simulation requires an even number of teams")

    use_buchholz = pairing == "buchholz"
    counters: Dict[str, Dict[str, int]] = {
        team.name: {"3-0": 0, "3-1": 0, "3-2": 0, "0-3": 0, "1-3": 0, "2-3": 0, "advance": 0, "eliminate": 0}
        for team in team_list
    }
    rng = random.Random(seed)
    joint_samples: List[Dict[str, Dict[str, object]]] = []

    for _ in range(simulations):
        states = {team.name: TeamState(team=team) for team in team_list}
        pairings = _opening_pairings_1v9(team_list) if use_buchholz else _opening_pairings(team_list)
        guard = 0
        while any(state.active for state in states.values()) and guard < 10:
            if guard > 0:
                pairings = _pair_active_buchholz(states) if use_buchholz else _pair_active(states)
            for team_a, team_b in pairings:
                state_a = states[team_a.name]
                state_b = states[team_b.name]
                if not state_a.active or not state_b.active:
                    continue
                best_of = 3 if _is_advancement_or_elimination(state_a, state_b) else 1
                probability_a = min(1.0, max(0.0, predictor(team_a, team_b, best_of, states)))
                winner, loser = (state_a, state_b) if rng.random() < probability_a else (state_b, state_a)
                winner.wins += 1
                loser.losses += 1
                state_a.opponents.add(team_b.name)
                state_b.opponents.add(team_a.name)
            guard += 1

        if collect_joint:
            joint_samples.append(_encode_joint_outcome(states))

        for name, state in states.items():
            record = f"{state.wins}-{state.losses}"
            if record in counters[name]:
                counters[name][record] += 1
            if state.wins >= 3:
                counters[name]["advance"] += 1
            if state.losses >= 3:
                counters[name]["eliminate"] += 1

    probabilities = {
        name: {key: value / simulations for key, value in counts.items()}
        for name, counts in counters.items()
    }
    return SwissSimulationResult(
        team_probabilities=probabilities,
        simulations=simulations,
        joint_samples=joint_samples,
    )


def _encode_joint_outcome(states: Dict[str, TeamState]) -> Dict[str, Dict[str, object]]:
    """Encode one finished bracket into a per-team category outcome vector."""
    outcome: Dict[str, Dict[str, object]] = {}
    for name, state in states.items():
        record = f"{state.wins}-{state.losses}"
        entry: Dict[str, object] = {
            "record": record,
            "advance": state.wins >= 3,
            "eliminate": state.losses >= 3,
        }
        for key in _RECORD_KEYS:
            entry[key] = record == key
        outcome[name] = entry
    return outcome


def _opening_pairings(teams: List[TeamSeed]) -> List[Tuple[TeamSeed, TeamSeed]]:
    ordered = sorted(teams, key=lambda team: team.seed)
    half = len(ordered) // 2
    return [(ordered[index], ordered[-(index + 1)]) for index in range(half)]


def _pair_active(states: Dict[str, TeamState]) -> List[Tuple[TeamSeed, TeamSeed]]:
    active = [state for state in states.values() if state.active]
    pairings: List[Tuple[TeamSeed, TeamSeed]] = []
    floater: List[TeamState] = []
    for _, bucket in _score_buckets(active):
        bucket_states = floater + bucket
        bucket_pairings, floater = _pair_score_bucket(bucket_states)
        pairings.extend(bucket_pairings)
    if len(floater) >= 2:
        bucket_pairings, floater = _pair_score_bucket(floater)
        pairings.extend(bucket_pairings)
    return pairings


def _score_buckets(active: List[TeamState]) -> List[Tuple[Tuple[int, int], List[TeamState]]]:
    buckets: Dict[Tuple[int, int], List[TeamState]] = {}
    for state in active:
        buckets.setdefault((state.wins, state.losses), []).append(state)
    ordered_keys = sorted(buckets, key=lambda record: (-record[0], record[1]))
    return [(key, sorted(buckets[key], key=lambda state: state.team.seed)) for key in ordered_keys]


def _pair_score_bucket(states: List[TeamState]) -> Tuple[List[Tuple[TeamSeed, TeamSeed]], List[TeamState]]:
    remaining = sorted(states, key=lambda state: state.team.seed)
    pairings: List[Tuple[TeamSeed, TeamSeed]] = []
    while len(remaining) >= 2:
        first = remaining.pop(0)
        opponent_index = _snake_opponent_index(first, remaining)
        second = remaining.pop(opponent_index)
        pairings.append((first.team, second.team))
    return pairings, remaining


def _snake_opponent_index(first: TeamState, candidates: List[TeamState]) -> int:
    for index in range(len(candidates) - 1, -1, -1):
        if candidates[index].team.name not in first.opponents:
            return index
    return len(candidates) - 1


def _is_advancement_or_elimination(state_a: TeamState, state_b: TeamState) -> bool:
    return state_a.wins == 2 or state_b.wins == 2 or state_a.losses == 2 or state_b.losses == 2


# ---------------------------------------------------------------------------
# Valve Major "Buchholz" pairing (official IEM Cologne 2026 Stage-1 rules).
# The MID-STAGE pairing algorithm (Buchholz re-rank, high-vs-low within a
# (wins,losses) bucket, no-rematch backtracking) is ported to match
# site/src/swissSim.js's engine. The ROUND-1 openings differ by design and are
# NOT required to be pair-for-pair identical: the Python MC uses the THEORETICAL
# Valve seeded form (seed_i vs seed_(i+half), i.e. 1v9, 2v10 … 8v16), whereas
# the front-end LIVE view seeds Round-1 from the REAL published draw constants
# (site/src/swissSim.js OPENING_PAIRINGS). Both are valid Round-1 inputs to the
# same downstream pairing engine; only the mid-stage algorithm is aligned. Kept
# separate from the legacy helpers above to preserve backward compatibility
# (default pairing="legacy"). See remainingConcerns: Python vs site Round-1.
# ---------------------------------------------------------------------------


def _opening_pairings_1v9(teams: List[TeamSeed]) -> List[Tuple[TeamSeed, TeamSeed]]:
    """Round-1 draw (theoretical Valve seeded form): seed_i vs seed_(i+half).

    16 teams -> 1v9, 2v10 … 8v16. This is the seeded *form* the MC assumes for
    Round-1; it is deliberately distinct from the front-end LIVE view, which
    seeds Round-1 from the real published draw constants (not a mechanical
    i-vs-i+half split). The two are NOT required to be pair-for-pair identical.
    """
    ordered = sorted(teams, key=lambda team: team.seed)
    half = len(ordered) // 2
    return [(ordered[index], ordered[index + half]) for index in range(half)]


def _buchholz(state: TeamState, states: Dict[str, TeamState]) -> int:
    """Difficulty score = Σ(opponent wins − opponent losses) over faced opponents."""
    total = 0
    for opponent_name in state.opponents:
        opponent = states.get(opponent_name)
        if opponent is not None:
            total += opponent.wins - opponent.losses
    return total


def _rank_bucket(bucket: List[TeamState], states: Dict[str, TeamState]) -> List[TeamState]:
    """Order a score bucket by Buchholz desc, then initial seed asc.

    Note: secondary tiebreak after Buchholz falls back to seed. The repo's rule
    sources do not specify a further (opponent-of-opponent) tiebreak, so this
    mirrors site/src/swissSim.js rankBucket — documented assumption.
    """
    return sorted(bucket, key=lambda state: (-_buchholz(state, states), state.team.seed))


def _pair_bucket_backtrack(
    ordered: List[TeamState],
) -> Tuple[List[Tuple[TeamSeed, TeamSeed]], List[TeamState]]:
    """Pair a ranked bucket high-vs-low with a best-effort no-rematch constraint.

    For even buckets, find a rematch-free perfect matching via deterministic,
    snake-biased backtracking (highest rank vs lowest available non-rematch).
    For odd buckets, or when NO rematch-free perfect matching exists, fall back
    to greedy snake pairing and carry the unpaired team forward as a floater
    (matching the front-end pairBucket implementation).

    NOTE: the fallback is "best-effort", not strictly rematch-free. When the
    bucket's opponent graph is so saturated that no rematch-free matching can
    exist (e.g. a complete-graph bucket), greedy snake WILL emit a rematch --
    this is mathematically unavoidable, not a bug. In the standard 16-team Major
    format every (wins,losses) bucket is even and solvable, so this fallback is
    not reached in practice (empirically 0 rematches over large MC runs).
    """
    count = len(ordered)
    if count < 2:
        return [], list(ordered)

    if count % 2 == 0:
        used = [False] * count
        pairs: List[Tuple[TeamSeed, TeamSeed]] = []

        def solve() -> bool:
            index = 0
            while index < count and used[index]:
                index += 1
            if index >= count:
                return True
            used[index] = True
            for other in range(count - 1, index, -1):
                if used[other] or ordered[other].team.name in ordered[index].opponents:
                    continue
                used[other] = True
                pairs.append((ordered[index].team, ordered[other].team))
                if solve():
                    return True
                pairs.pop()
                used[other] = False
            used[index] = False
            return False

        if solve():
            return pairs, []

    # Odd bucket, or no rematch-free matching exists: greedy snake + floater.
    remaining = list(ordered)
    pairings: List[Tuple[TeamSeed, TeamSeed]] = []
    while len(remaining) >= 2:
        first = remaining.pop(0)
        opponent_index = _snake_opponent_index(first, remaining)
        second = remaining.pop(opponent_index)
        pairings.append((first.team, second.team))
    return pairings, remaining


def _pair_active_buchholz(states: Dict[str, TeamState]) -> List[Tuple[TeamSeed, TeamSeed]]:
    """Pair all active teams under Valve Buchholz rules with no rematches."""
    active = [state for state in states.values() if state.active]
    buckets: Dict[Tuple[int, int], List[TeamState]] = {}
    for state in active:
        buckets.setdefault((state.wins, state.losses), []).append(state)
    # Bucket order: more wins first, then fewer losses (site pairNextRound key).
    ordered_keys = sorted(buckets, key=lambda record: (-record[0], record[1]))

    pairings: List[Tuple[TeamSeed, TeamSeed]] = []
    floater: List[TeamState] = []
    for key in ordered_keys:
        ranked = _rank_bucket(floater + buckets[key], states)
        bucket_pairings, floater = _pair_bucket_backtrack(ranked)
        pairings.extend(bucket_pairings)
    if len(floater) >= 2:
        ranked = _rank_bucket(floater, states)
        bucket_pairings, floater = _pair_bucket_backtrack(ranked)
        pairings.extend(bucket_pairings)
    return pairings

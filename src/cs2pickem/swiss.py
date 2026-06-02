from __future__ import annotations

import random
from dataclasses import dataclass
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


Predictor = Callable[[TeamSeed, TeamSeed, int, Dict[str, TeamState]], float]


def simulate_swiss(
    teams: Iterable[TeamSeed],
    predictor: Predictor,
    simulations: int = 100000,
    seed: int = 13,
) -> SwissSimulationResult:
    team_list = sorted(list(teams), key=lambda team: team.seed)
    if len(team_list) % 2 != 0:
        raise ValueError("Swiss simulation requires an even number of teams")

    counters: Dict[str, Dict[str, int]] = {
        team.name: {"3-0": 0, "3-1": 0, "3-2": 0, "0-3": 0, "1-3": 0, "2-3": 0, "advance": 0, "eliminate": 0}
        for team in team_list
    }
    rng = random.Random(seed)

    for _ in range(simulations):
        states = {team.name: TeamState(team=team) for team in team_list}
        pairings = _opening_pairings(team_list)
        guard = 0
        while any(state.active for state in states.values()) and guard < 10:
            if guard > 0:
                pairings = _pair_active(states)
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
    return SwissSimulationResult(team_probabilities=probabilities, simulations=simulations)


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

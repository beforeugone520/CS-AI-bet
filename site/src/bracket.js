export function emptyBracketState(bracket) {
  const originalBracket = cloneBracket(bracket);
  const matches = {};
  for (const round of ["quarterfinals", "semifinals", "final"]) {
    for (const match of bracket[round] || []) {
      matches[match.id] = { ...match, round, winner: match.winner || null };
    }
  }
  return { originalBracket, matches, champion: null, history: [] };
}

export function applyBracketWinner(state, matchId, winner) {
  const matches = cloneMatches(state.matches);
  const match = matches[matchId];
  if (!match) {
    throw new Error("match not found");
  }
  if (winner !== match.team1 && winner !== match.team2) {
    throw new Error("winner must be one of the match teams");
  }
  match.winner = winner;
  let champion = state.champion;
  if (match.nextMatchId) {
    matches[match.nextMatchId] = {
      ...matches[match.nextMatchId],
      [match.nextSlot]: winner
    };
  } else {
    champion = winner;
  }
  return {
    originalBracket: cloneBracket(state.originalBracket),
    matches,
    champion,
    history: state.history.concat([{ matchId, winner }])
  };
}

export function undoBracket(state) {
  if (state.history.length === 0) {
    return state;
  }
  const history = state.history.slice(0, -1);
  let replay = emptyBracketState(state.originalBracket);
  for (const entry of history) {
    replay = applyBracketWinner(replay, entry.matchId, entry.winner);
  }
  return replay;
}

export function resetBracket(bracket) {
  return emptyBracketState(bracket);
}

function cloneBracket(bracket) {
  const cloned = {};
  for (const round of ["quarterfinals", "semifinals", "final"]) {
    cloned[round] = (bracket[round] || []).map((match) => ({ ...match }));
  }
  return cloned;
}

function cloneMatches(matches) {
  const cloned = {};
  for (const [id, match] of Object.entries(matches)) {
    cloned[id] = { ...match };
  }
  return cloned;
}

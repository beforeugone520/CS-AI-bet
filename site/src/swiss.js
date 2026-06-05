export function resetSwissState(standings) {
  const records = recordsFromStandings(standings);
  return {
    initialRecords: cloneRecords(records),
    records,
    history: []
  };
}

export function applySwissWinner(state, fixture, winner) {
  if (winner !== fixture.team1 && winner !== fixture.team2) {
    throw new Error("winner must be one of the fixture teams");
  }
  const loser = winner === fixture.team1 ? fixture.team2 : fixture.team1;
  const records = cloneRecords(state.records);
  records[winner] = bump(records[winner], 1, 0);
  records[loser] = bump(records[loser], 0, 1);
  return {
    initialRecords: cloneRecords(state.initialRecords || state.records),
    records,
    history: state.history.concat([{ fixture, winner, loser }])
  };
}

export function undoSwiss(state) {
  if (state.history.length === 0) {
    return state;
  }
  const history = state.history.slice(0, -1);
  let replay = {
    initialRecords: cloneRecords(state.initialRecords),
    records: cloneRecords(state.initialRecords),
    history: []
  };
  for (const entry of history) {
    replay = applySwissWinner(replay, entry.fixture, entry.winner);
  }
  return replay;
}

export function recordStatus(wins, losses) {
  if (wins >= 3) return "advanced";
  if (losses >= 3) return "eliminated";
  return "alive";
}

function bump(record, winDelta, lossDelta) {
  if (!record) {
    throw new Error("record missing for team");
  }
  const wins = Number(record.wins) + winDelta;
  const losses = Number(record.losses) + lossDelta;
  return { ...record, wins, losses, status: recordStatus(wins, losses) };
}

function cloneRecords(records) {
  const cloned = {};
  for (const [team, record] of Object.entries(records)) {
    cloned[team] = { ...record };
  }
  return cloned;
}

function recordsFromStandings(rows) {
  const records = {};
  for (const row of rows) {
    const wins = Number(row.wins || 0);
    const losses = Number(row.losses || 0);
    records[row.team] = {
      team: row.team,
      wins,
      losses,
      status: row.status || recordStatus(wins, losses)
    };
  }
  return records;
}

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
  const key = fixtureKey(fixture);
  const entry = { fixture, key, winner, loser };
  const history = state.history.filter((item) => (item.key || fixtureKey(item.fixture)) !== key).concat([entry]);
  return replaySwissHistory(state.initialRecords || state.records, history);
}

export function undoSwiss(state) {
  if (state.history.length === 0) {
    return state;
  }
  const history = state.history.slice(0, -1);
  return replaySwissHistory(state.initialRecords || state.records, history);
}

export function clearSwissSelections(state) {
  return {
    initialRecords: cloneRecords(state.initialRecords || state.records),
    records: cloneRecords(state.initialRecords || state.records),
    history: []
  };
}

export function groupSwissRecords(records) {
  const rows = Object.values(records).map((row) => ({ ...row })).sort(compareRecords);
  return {
    advanced: rows.filter((row) => row.status === "advanced"),
    live: rows.filter((row) => row.status === "alive"),
    eliminated: rows.filter((row) => row.status === "eliminated")
  };
}

export function recordStatus(wins, losses) {
  if (wins >= 3) return "advanced";
  if (losses >= 3) return "eliminated";
  return "alive";
}

export function fixtureKey(fixture) {
  const round = fixture.swiss_round || fixture.round || "round";
  const teams = [fixture.team1, fixture.team2].sort().join("_");
  return String(fixture.id || fixture.source_match_url || `${round}:${teams}`);
}

function replaySwissHistory(initialRecords, history) {
  const records = cloneRecords(initialRecords);
  for (const entry of history) {
    records[entry.winner] = bump(records[entry.winner], 1, 0);
    records[entry.loser] = bump(records[entry.loser], 0, 1);
  }
  return {
    initialRecords: cloneRecords(initialRecords),
    records,
    history: history.map((entry) => ({ ...entry, fixture: { ...entry.fixture } }))
  };
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

function compareRecords(a, b) {
  return Number(b.wins) - Number(a.wins) || Number(a.losses) - Number(b.losses) || String(a.team).localeCompare(String(b.team));
}

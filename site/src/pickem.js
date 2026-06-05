export function classifyPickem(pickemPayload, records) {
  return pickemPayload.picks.map((pick) => {
    const record = records[pick.team];
    return {
      ...pick,
      wins: record ? Number(record.wins) : null,
      losses: record ? Number(record.losses) : null,
      status: pickStatus(pick.category, record)
    };
  });
}

export function summarizePickem(rows) {
  const summary = { locked: 0, alive: 0, broken: 0, missing: 0 };
  for (const row of rows) {
    summary[row.status] = (summary[row.status] || 0) + 1;
  }
  return summary;
}

export function pickStatus(category, record) {
  if (!record) return "missing";
  const wins = Number(record.wins);
  const losses = Number(record.losses);
  if (category === "3-0") {
    if (wins >= 3 && losses === 0) return "locked";
    if (losses > 0 || (wins >= 3 && losses > 0)) return "broken";
    return "alive";
  }
  if (category === "advance") {
    if (wins >= 3) return "locked";
    if (losses >= 3) return "broken";
    return "alive";
  }
  if (category === "0-3") {
    if (losses >= 3 && wins === 0) return "locked";
    if (wins > 0 || (losses >= 3 && wins > 0)) return "broken";
    return "alive";
  }
  return "missing";
}

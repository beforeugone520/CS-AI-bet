/* ============================================================================
   Pure logic for the PREDICT selector UX layer (hover-trace, FLIP, conflict).
   No `document` access at module top-level — unit-testable with plain imports.
   ========================================================================== */

// Which match cards a team appears in, plus its terminal-column status.
export function traceTeam(bracket, team) {
  const matchKeys = new Set();
  if (!bracket || !team) return { matchKeys, terminal: "alive" };
  for (const rd of bracket.rounds || []) {
    for (const m of rd.matches || []) {
      if (m.team1 === team || m.team2 === team) matchKeys.add(m.key);
    }
  }
  let terminal = "alive";
  for (const s of bracket.standings || []) {
    if (s.team === team) { terminal = s.status; break; }
  }
  return { matchKeys, terminal };
}

// Geometric FLIP diff: old position minus new position, per stable key.
// Returns only keys that exist in both snapshots and actually moved.
export function computeFlipTransforms(oldRects, newRects) {
  const out = [];
  for (const key of Object.keys(newRects || {})) {
    const a = (oldRects || {})[key];
    const b = newRects[key];
    if (!a || !b) continue;                 // unmatched -> fade, not flip
    const dx = a.left - b.left;
    const dy = a.top - b.top;
    if (dx === 0 && dy === 0) continue;
    out.push({ key, dx, dy });
  }
  return out;
}

// Picks that no longer reference a live matchup, or whose winner is no longer
// a participant of it — mirrors main.js's inline prune so the toast can name them.
export function diffPrunedPicks(prevPicks, bracket) {
  const teamsByKey = new Map();
  for (const rd of (bracket && bracket.rounds) || []) {
    for (const m of rd.matches || []) teamsByKey.set(m.key, [m.team1, m.team2]);
  }
  const pruned = [];
  for (const [k, v] of Object.entries(prevPicks || {})) {
    const teams = teamsByKey.get(k);
    if (!teams || !teams.includes(v)) {
      pruned.push({ key: k, team: v, label: String(k).split(" :: ").join(" vs ") });
    }
  }
  return pruned;
}

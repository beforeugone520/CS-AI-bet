/* ============================================================================
   Full interactive Swiss-stage simulator (majors.im-style).

   HYBRID model: the real event matchups (from exported static data) are the
   ground truth. While the user's picks match the real winners, the board
   reproduces the EXACT real bracket round-by-round. The moment a pick diverges
   from reality, subsequent rounds are paired by the ported swiss.py engine
   (seed + Buchholz bucket pairing, rematch avoidance, advance@3W / eliminate@3L).

   Deterministic: the whole bracket is rebuilt from a { matchKey: winner } map,
   so editing any pick re-pairs everything downstream.
   ========================================================================== */

export const SWISS_SEEDS = {
  "GamerLegion": 1, "B8": 2, "BetBoom": 3, "MIBR": 4,
  "HEROIC": 5, "Lynn Vision": 6, "BIG": 7, "TYLOO": 8,
  "SINNERS": 9, "M80": 10, "Liquid": 11, "Sharks": 12,
  "NRG": 13, "Gaimin Gladiators": 14, "THUNDER dOWNUNDER": 15, "FlyQuest": 16
};

// Real Round-1 (0-0) draw — fallback when no exported data is supplied
export const OPENING_PAIRINGS = [
  ["M80", "Lynn Vision"], ["SINNERS", "FlyQuest"], ["B8", "TYLOO"],
  ["MIBR", "THUNDER dOWNUNDER"], ["GamerLegion", "NRG"], ["HEROIC", "Sharks"],
  ["BetBoom", "Gaimin Gladiators"], ["BIG", "Liquid"]
];

export const ADVANCE_AT = 3;
export const ELIMINATE_AT = 3;
export const TOTAL_ROUNDS = 5;

export function matchKey(a, b) {
  return [String(a), String(b)].sort().join(" :: ");
}

/* ---------- ported swiss.py pairing engine (for divergent branches) ---------- */
function freshStates(seeds) {
  const states = {};
  for (const name of Object.keys(seeds)) {
    states[name] = { name, seed: seeds[name], wins: 0, losses: 0, opponents: new Set(), buchholz: 0 };
  }
  return states;
}

function isDecider(s1, s2) {
  return s1.wins === ADVANCE_AT - 1 || s2.wins === ADVANCE_AT - 1 ||
    s1.losses === ELIMINATE_AT - 1 || s2.losses === ELIMINATE_AT - 1;
}

function updateBuchholz(states) {
  for (const s of Object.values(states)) {
    let b = 0;
    for (const opp of s.opponents) {
      const o = states[opp];
      if (o) b += (o.wins - o.losses);
    }
    s.buchholz = b;
  }
}

function rankBucket(bucket) {
  return bucket.slice().sort((a, b) => (b.buchholz - a.buchholz) || (a.seed - b.seed));
}

function pairBucket(ordered) {
  const n = ordered.length;
  if (n < 2) return { pairings: [], floater: ordered.slice() };

  // Even bucket: find a rematch-FREE perfect matching via deterministic,
  // snake-biased backtracking (highest rank vs lowest available non-rematch).
  // This fixes the greedy pairer's avoidable rematches.
  if (n % 2 === 0) {
    const used = new Array(n).fill(false);
    const pairs = [];
    const solve = () => {
      let i = 0;
      while (i < n && used[i]) i += 1;
      if (i >= n) return true;
      used[i] = true;
      for (let j = n - 1; j > i; j -= 1) {
        if (used[j] || ordered[i].opponents.has(ordered[j].name)) continue;
        used[j] = true;
        pairs.push([ordered[i], ordered[j]]);
        if (solve()) return true;
        pairs.pop();
        used[j] = false;
      }
      used[i] = false;
      return false;
    };
    if (solve()) return { pairings: pairs, floater: [] };
  }

  // Odd bucket, or no rematch-free matching exists: greedy snake pairing,
  // carrying the unpaired team to the next bucket (matches swiss.py).
  const rem = ordered.slice();
  const pairings = [];
  while (rem.length >= 2) {
    const first = rem.shift();
    let idx = -1;
    for (let i = rem.length - 1; i >= 0; i--) {
      if (!first.opponents.has(rem[i].name)) { idx = i; break; }
    }
    if (idx < 0) idx = rem.length - 1;
    pairings.push([first, rem.splice(idx, 1)[0]]);
  }
  return { pairings, floater: rem };
}

function pairNextRound(states) {
  const active = Object.values(states).filter((s) => s.wins < ADVANCE_AT && s.losses < ELIMINATE_AT);
  const buckets = {};
  for (const s of active) {
    const k = `${s.wins}-${s.losses}`;
    (buckets[k] = buckets[k] || []).push(s);
  }
  const keys = Object.keys(buckets).sort((a, b) => {
    const [aw, al] = a.split("-").map(Number);
    const [bw, bl] = b.split("-").map(Number);
    return (bw - aw) || (al - bl);
  });
  const pairs = [];
  let floater = [];
  for (const k of keys) {
    const { pairings, floater: rem } = pairBucket(rankBucket(floater.concat(buckets[k])));
    pairs.push(...pairings);
    floater = rem;
  }
  if (floater.length >= 2) pairs.push(...pairBucket(rankBucket(floater)).pairings);
  return pairs.map(([a, b]) => [a.name, b.name]);
}

/* ---------- hybrid bracket builder ---------- */
/**
 * @param {Object} picks   { matchKey: winnerName }
 * @param {Object} [opts]   { seeds, realByRound: {1:[{team1,team2}],...}, realWinners: {matchKey:winner} }
 */
export function buildSwiss(picks = {}, opts = {}) {
  const seeds = opts.seeds || SWISS_SEEDS;
  const realByRound = opts.realByRound || null;
  const realWinners = opts.realWinners || {};
  const realScores = opts.realScores || {};
  const states = freshStates(seeds);
  const rounds = [];
  const validKeys = new Set();
  let aligned = true; // picks so far reproduce reality

  let pairList = (realByRound && realByRound[1])
    ? realByRound[1].map((m) => [m.team1, m.team2])
    : OPENING_PAIRINGS.map((p) => p.slice());
  let roundNum = 1;

  while (pairList.length && roundNum <= TOTAL_ROUNDS) {
    const fromReal = aligned && !!(realByRound && realByRound[roundNum]);
    const matches = pairList.map(([t1, t2]) => {
      const s1 = states[t1], s2 = states[t2];
      const key = matchKey(t1, t2);
      validKeys.add(key);
      const picked = picks[key];
      const winner = picked === t1 || picked === t2 ? picked : null;
      return {
        key, round: roundNum, team1: t1, team2: t2,
        bo: isDecider(s1, s2) ? 3 : 1,
        record: `${s1.wins}-${s1.losses}`,
        record2: `${s2.wins}-${s2.losses}`,
        decider: isDecider(s1, s2),
        fromReal,
        realWinner: fromReal ? (realWinners[key] ?? null) : null,
        realScore: fromReal ? (realScores[key] ?? null) : null,
        winner
      };
    });
    rounds.push({ round: roundNum, matches, fromReal });

    const allDecided = matches.every((m) => m.winner);
    for (const m of matches) {
      if (!m.winner) continue;
      const w = m.winner;
      const l = m.team1 === w ? m.team2 : m.team1;
      states[w].wins += 1;
      states[l].losses += 1;
      states[w].opponents.add(l);
      states[l].opponents.add(w);
      // alignment: a real R1-4 matchup must be won by the real winner to stay aligned.
      // R5 has no real winner yet (realWinner null) but is the last round, so it never gates a next round.
      // Diverge only when a KNOWN real winner is contradicted; unknown winners
      // (R5 not played, or partially-populated data) stay consistent.
      const rw = realWinners[m.key];
      if (rw !== undefined && rw !== w) aligned = false;
    }

    if (!allDecided || roundNum >= TOTAL_ROUNDS) break;
    updateBuchholz(states);
    pairList = (aligned && realByRound && realByRound[roundNum + 1])
      ? realByRound[roundNum + 1].map((m) => [m.team1, m.team2])
      : pairNextRound(states);
    roundNum += 1;
  }

  updateBuchholz(states);
  const standings = Object.values(states).map((s) => ({
    team: s.name, seed: s.seed, wins: s.wins, losses: s.losses, buchholz: s.buchholz,
    record: `${s.wins}-${s.losses}`,
    status: s.wins >= ADVANCE_AT ? "advanced" : (s.losses >= ELIMINATE_AT ? "eliminated" : "alive")
  })).sort(compareStanding);

  const advancedCount = standings.filter((r) => r.status === "advanced").length;
  return { rounds, states, standings, picks, validKeys, complete: advancedCount >= 8, aligned };
}

/* ---------- data + pick helpers ---------- */
export function realRoundsFromStage(stage) {
  const byRound = {};
  const winners = {};
  const scores = {};
  for (const r of stage.results || []) {
    const k = Number(r.round);
    if (!k) continue;
    (byRound[k] = byRound[k] || []).push({ team1: r.team1, team2: r.team2 });
    const key = matchKey(r.team1, r.team2);
    if (r.winner) winners[key] = r.winner;
    if (r.match_score) scores[key] = r.match_score;
  }
  for (const f of stage.fixtures || []) {
    const k = Number(f.swiss_round || f.round);
    if (!k) continue;
    (byRound[k] = byRound[k] || []).push({ team1: f.team1, team2: f.team2 });
  }
  return { realByRound: byRound, realWinners: winners, realScores: scores };
}

export function prunePicks(picks, validKeys) {
  const next = {};
  for (const [k, v] of Object.entries(picks)) if (validKeys.has(k)) next[k] = v;
  return next;
}

export function picksFromResults(results) {
  const picks = {};
  for (const r of results || []) {
    if (r && r.team1 && r.team2 && r.winner) picks[matchKey(r.team1, r.team2)] = r.winner;
  }
  return picks;
}

export function countPicks(picks, validKeys) {
  let n = 0;
  for (const k of Object.keys(picks)) if (!validKeys || validKeys.has(k)) n += 1;
  return n;
}

function compareStanding(a, b) {
  return (b.wins - a.wins) || (a.losses - b.losses) || (b.buchholz - a.buchholz) || (a.seed - b.seed);
}

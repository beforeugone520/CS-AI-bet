import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import {
  buildSwiss, picksFromResults, realRoundsFromStage, prunePicks, matchKey, countPicks
} from "../src/swissSim.js";

const stage = JSON.parse(fs.readFileSync(new URL("../data/stages/stage-1.json", import.meta.url), "utf8"));
const realData = realRoundsFromStage(stage);

test("blank slate shows only Round 1 (8 opening matches)", () => {
  const b = buildSwiss({}, realData);
  assert.equal(b.rounds.length, 1);
  assert.equal(b.rounds[0].matches.length, 8);
  assert.equal(b.standings.filter((s) => s.status !== "alive").length, 0);
});

test("real results reproduce the exact real bracket through Round 5", () => {
  const picks = picksFromResults(stage.results);
  const b = buildSwiss(picks, realData);
  assert.ok(b.aligned, "stays aligned with reality");
  const genR5 = new Set((b.rounds.find((r) => r.round === 5)?.matches || []).map((m) => m.key));
  const realR5 = new Set((stage.fixtures || []).map((f) => matchKey(f.team1, f.team2)));
  assert.equal(genR5.size, realR5.size);
  for (const k of realR5) assert.ok(genR5.has(k), `real R5 matchup present: ${k}`);
  // BO progression: R1/R2 BO1, R5 BO3 deciders
  assert.ok(b.rounds[0].matches.every((m) => m.bo === 1));
  assert.ok(b.rounds.find((r) => r.round === 5).matches.every((m) => m.bo === 3));
});

test("post-Round-4 standings: 5 advanced / 6 alive / 5 eliminated", () => {
  const b = buildSwiss(picksFromResults(stage.results), realData);
  const by = (s) => b.standings.filter((r) => r.status === s).length;
  assert.equal(by("advanced"), 5);
  assert.equal(by("alive"), 6);
  assert.equal(by("eliminated"), 5);
});

test("completing Round 5 yields 8 advanced and 8 eliminated", () => {
  const picks = picksFromResults(stage.results);
  let b = buildSwiss(picks, realData);
  for (const m of b.rounds.find((r) => r.round === 5).matches) picks[m.key] = m.team1;
  b = buildSwiss(picks, realData);
  assert.equal(b.standings.filter((s) => s.status === "advanced").length, 8);
  assert.equal(b.standings.filter((s) => s.status === "eliminated").length, 8);
  assert.ok(b.complete);
});

test("diverging from reality re-pairs via the engine and stays structurally sane", () => {
  const picks = picksFromResults(stage.results);
  picks[matchKey("SINNERS", "FlyQuest")] = "SINNERS"; // reality: FlyQuest won
  const b = buildSwiss(picks, realData);
  assert.equal(b.aligned, false);
  for (const rd of b.rounds) {
    const seen = new Set();
    for (const m of rd.matches) {
      assert.ok(!seen.has(m.team1) && !seen.has(m.team2), `no team appears twice in round ${rd.round}`);
      seen.add(m.team1); seen.add(m.team2);
    }
  }
});

test("no rematches are generated", () => {
  const picks = picksFromResults(stage.results);
  picks[matchKey("BIG", "Liquid")] = "BIG"; // flip an opener
  const b = buildSwiss(picks, realData);
  const played = new Set();
  for (const rd of b.rounds) {
    for (const m of rd.matches.filter((x) => x.winner)) {
      assert.ok(!played.has(m.key), `no rematch: ${m.key}`);
      played.add(m.key);
    }
  }
});

test("prunePicks drops keys not in the current bracket", () => {
  const b = buildSwiss({}, realData);
  const picks = { [matchKey("M80", "Lynn Vision")]: "M80", "Ghost :: Team": "Ghost" };
  const pruned = prunePicks(picks, b.validKeys);
  assert.ok(pruned[matchKey("M80", "Lynn Vision")]);
  assert.equal(pruned["Ghost :: Team"], undefined);
});

test("matchKey is order-independent", () => {
  assert.equal(matchKey("A", "B"), matchKey("B", "A"));
  assert.equal(countPicks({ [matchKey("A", "B")]: "A" }), 1);
});

test("fuzz: 300 random full brackets contain no rematches and always complete 8/8", () => {
  let seed = 0x2f6b9d1;
  const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  for (let iter = 0; iter < 300; iter += 1) {
    const picks = {};
    let b = buildSwiss(picks, realData);
    let guard = 0;
    while (!b.complete && guard < 12) {
      const rd = b.rounds.find((r) => r.matches.some((m) => !m.winner));
      if (!rd) break;
      for (const m of rd.matches) {
        if (!m.winner) picks[m.key] = rand() < 0.5 ? m.team1 : m.team2;
      }
      b = buildSwiss(picks, realData);
      guard += 1;
    }
    const played = [];
    for (const r of b.rounds) for (const m of r.matches) if (m.winner) played.push(m.key);
    assert.equal(new Set(played).size, played.length, `iter ${iter}: an avoidable rematch occurred`);
    assert.equal(b.standings.filter((s) => s.status === "advanced").length, 8, `iter ${iter}: must reach 8 advanced`);
    assert.equal(b.standings.filter((s) => s.status === "eliminated").length, 8, `iter ${iter}: must reach 8 eliminated`);
  }
});

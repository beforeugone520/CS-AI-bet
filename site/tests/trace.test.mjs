import { test } from "node:test";
import assert from "node:assert/strict";
import { buildSwiss } from "../src/swissSim.js";
import { traceTeam, computeFlipTransforms, diffPrunedPicks } from "../src/trace.js";

test("traceTeam collects every match a team plays, all containing that team", () => {
  const b = buildSwiss({});
  const { matchKeys } = traceTeam(b, "M80");
  assert.ok(matchKeys.size >= 1);
  for (const k of matchKeys) assert.ok(k.includes("M80"));
});

test("traceTeam reports terminal status from standings", () => {
  const b = buildSwiss({});
  assert.equal(traceTeam(b, "M80").terminal, "alive");
});

test("traceTeam is robust to null bracket and unknown team", () => {
  const empty = traceTeam(null, "X");
  assert.equal(empty.matchKeys.size, 0);
  assert.equal(empty.terminal, "alive");
  const b = buildSwiss({});
  const unknown = traceTeam(b, "NotATeam");
  assert.equal(unknown.matchKeys.size, 0);
  assert.equal(unknown.terminal, "alive");
});

test("computeFlipTransforms returns dx/dy only for moved keys", () => {
  const oldR = { a: { left: 0, top: 0 }, b: { left: 10, top: 10 } };
  const newR = { a: { left: 0, top: 0 }, b: { left: 40, top: 30 } };
  const out = computeFlipTransforms(oldR, newR);
  assert.equal(out.length, 1);
  assert.deepEqual(out[0], { key: "b", dx: -30, dy: -20 });
});

test("computeFlipTransforms skips unmatched keys and null input", () => {
  assert.equal(computeFlipTransforms({ a: { left: 0, top: 0 } }, { b: { left: 5, top: 5 } }).length, 0);
  assert.equal(computeFlipTransforms(null, null).length, 0);
});

test("diffPrunedPicks flags picks whose matchup no longer exists", () => {
  const b = buildSwiss({});
  const dropped = diffPrunedPicks({ "ZZZ :: YYY": "ZZZ" }, b);
  assert.equal(dropped.length, 1);
  assert.equal(dropped[0].key, "ZZZ :: YYY");
  assert.ok(dropped[0].label.includes("vs"));
});

test("diffPrunedPicks keeps a valid round-1 pick", () => {
  const b = buildSwiss({});
  const m = b.rounds[0].matches[0];
  assert.equal(diffPrunedPicks({ [m.key]: m.team1 }, b).length, 0);
});

test("diffPrunedPicks flags a pick whose winner is not a participant", () => {
  const b = buildSwiss({});
  const m = b.rounds[0].matches[0];
  assert.equal(diffPrunedPicks({ [m.key]: "Wrong" }, b).length, 1);
});

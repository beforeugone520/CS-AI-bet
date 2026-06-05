import assert from "node:assert/strict";
import test from "node:test";
import { applySwissWinner, resetSwissState, undoSwiss } from "../src/swiss.js";

const standings = [
  { team: "BIG", wins: 2, losses: 2, status: "alive" },
  { team: "NRG", wins: 2, losses: 2, status: "alive" },
  { team: "M80", wins: 3, losses: 1, status: "advanced" }
];

test("applySwissWinner advances winner and eliminates loser at 2-2", () => {
  const state = resetSwissState(standings);
  const next = applySwissWinner(state, { team1: "NRG", team2: "BIG" }, "BIG");

  assert.equal(next.records.BIG.wins, 3);
  assert.equal(next.records.BIG.status, "advanced");
  assert.equal(next.records.NRG.losses, 3);
  assert.equal(next.records.NRG.status, "eliminated");
  assert.equal(next.history.length, 1);
});

test("applySwissWinner does not mutate previous state", () => {
  const state = resetSwissState(standings);
  const next = applySwissWinner(state, { team1: "NRG", team2: "BIG" }, "BIG");

  assert.equal(state.records.BIG.wins, 2);
  assert.equal(next.records.BIG.wins, 3);
});

test("undoSwiss removes only the latest simulated result", () => {
  const state = resetSwissState(standings);
  const first = applySwissWinner(state, { team1: "NRG", team2: "BIG" }, "BIG");
  const second = applySwissWinner(first, { team1: "M80", team2: "BIG" }, "M80");
  const undone = undoSwiss(second);

  assert.equal(undone.history.length, 1);
  assert.equal(undone.records.BIG.wins, 3);
  assert.equal(undone.records.NRG.status, "eliminated");
  assert.equal(undone.records.M80.wins, 3);
});

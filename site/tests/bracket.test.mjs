import assert from "node:assert/strict";
import test from "node:test";
import { applyBracketWinner, emptyBracketState, resetBracket, undoBracket } from "../src/bracket.js";

const bracket = {
  quarterfinals: [
    { id: "qf-1", team1: "Alpha", team2: "Bravo", nextMatchId: "sf-1", nextSlot: "team1" }
  ],
  semifinals: [{ id: "sf-1", team1: null, team2: "Charlie", nextMatchId: "final", nextSlot: "team1" }],
  final: [{ id: "final", team1: null, team2: null, nextMatchId: null, nextSlot: null }]
};

test("applyBracketWinner advances quarterfinal winner to semifinal slot", () => {
  const state = emptyBracketState(bracket);

  const next = applyBracketWinner(state, "qf-1", "Alpha");
  assert.equal(next.matches["qf-1"].winner, "Alpha");
  assert.equal(next.matches["sf-1"].team1, "Alpha");
});

test("applyBracketWinner sets champion when final winner is chosen", () => {
  const state = emptyBracketState({
    quarterfinals: [],
    semifinals: [],
    final: [{ id: "final", team1: "Alpha", team2: "Charlie", nextMatchId: null, nextSlot: null }]
  });

  const next = applyBracketWinner(state, "final", "Charlie");
  assert.equal(next.champion, "Charlie");
});

test("undoBracket removes only the latest simulated winner", () => {
  const first = applyBracketWinner(emptyBracketState(bracket), "qf-1", "Alpha");
  const second = applyBracketWinner(first, "sf-1", "Alpha");
  const undone = undoBracket(second);

  assert.equal(undone.history.length, 1);
  assert.equal(undone.matches["qf-1"].winner, "Alpha");
  assert.equal(undone.matches["sf-1"].winner, null);
  assert.equal(undone.matches["final"].team1, null);
});

test("resetBracket clears all simulated winners", () => {
  const selected = applyBracketWinner(emptyBracketState(bracket), "qf-1", "Alpha");
  const reset = resetBracket(selected.originalBracket);

  assert.equal(reset.history.length, 0);
  assert.equal(reset.matches["qf-1"].winner, null);
  assert.equal(reset.matches["sf-1"].team1, null);
});

import assert from "node:assert/strict";
import test from "node:test";
import { classifyPickem, summarizePickem } from "../src/pickem.js";

const pickems = {
  picks: [
    { category: "advance", team: "BIG" },
    { category: "advance", team: "TYLOO" },
    { category: "0-3", team: "Gaimin Gladiators" },
    { category: "3-0", team: "MIBR" }
  ]
};

test("classifyPickem tracks locked alive and broken states", () => {
  const records = {
    BIG: { wins: 3, losses: 2, status: "advanced" },
    TYLOO: { wins: 2, losses: 2, status: "alive" },
    "Gaimin Gladiators": { wins: 0, losses: 3, status: "eliminated" },
    MIBR: { wins: 3, losses: 1, status: "advanced" }
  };

  const rows = classifyPickem(pickems, records);
  assert.equal(rows.find((row) => row.team === "BIG").status, "locked");
  assert.equal(rows.find((row) => row.team === "TYLOO").status, "alive");
  assert.equal(rows.find((row) => row.team === "Gaimin Gladiators").status, "locked");
  assert.equal(rows.find((row) => row.team === "MIBR").status, "broken");
});

test("summarizePickem counts statuses", () => {
  const rows = [
    { status: "locked" },
    { status: "alive" },
    { status: "broken" },
    { status: "locked" }
  ];

  assert.deepEqual(summarizePickem(rows), { locked: 2, alive: 1, broken: 1, missing: 0 });
});

test("summarizePickem includes missing status when record is absent", () => {
  const rows = classifyPickem({ picks: [{ category: "advance", team: "Unknown" }] }, {});
  assert.deepEqual(summarizePickem(rows), { locked: 0, alive: 0, broken: 0, missing: 1 });
});

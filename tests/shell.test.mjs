import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

test("public shell exposes only the current Major simulator", () => {
  const html = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");

  assert.match(html, /IEM Cologne Major 2026 Simulator/);
  assert.match(html, /assets\/events\/iem\.png/);
  assert.doesNotMatch(html, /AI Desk/);
  assert.doesNotMatch(html, /Model Lab/);
  assert.doesNotMatch(html, /Overview/);
});

test("current Major logo assets are bundled with the static site", () => {
  const teams = ["b8", "betb", "big", "fly", "gg", "gl", "hero", "liqu", "lvg", "m80", "mibr", "nrg", "shks", "sinn", "tdu", "tylo"];

  assert.ok(fs.existsSync(new URL("../assets/events/iem.png", import.meta.url)));
  for (const team of teams) {
    assert.ok(fs.existsSync(new URL(`../assets/teams/${team}.png`, import.meta.url)), `${team}.png should exist`);
  }
});

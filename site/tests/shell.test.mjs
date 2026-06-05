import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

test("public shell exposes only the current Major simulator", () => {
  const html = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");

  assert.match(html, /IEM Cologne Major 2026 Simulator/);
  assert.doesNotMatch(html, /AI Desk/);
  assert.doesNotMatch(html, /Model Lab/);
  assert.doesNotMatch(html, /Overview/);
});

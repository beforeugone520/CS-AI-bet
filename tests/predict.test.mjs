import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import { buildSwiss, picksFromResults, realRoundsFromStage, matchKey, countPicks } from "../src/swissSim.js";
import { renderPredictHtml } from "../src/renderPredict.js";

const stage = JSON.parse(fs.readFileSync(new URL("../data/stages/stage-1.json", import.meta.url), "utf8"));
const realData = realRoundsFromStage(stage);

test("predict board renders the majors.im-style interactive bracket", () => {
  const picks = picksFromResults(stage.results);
  const b = buildSwiss(picks, realData);
  const html = renderPredictHtml(b, null, "simple", { pickCount: countPicks(picks, b.validKeys) });

  assert.match(html, /predict-board/);
  assert.match(html, /mode-switch/);
  assert.match(html, /data-swiss-mode="predict"/);
  assert.match(html, /data-swiss-mode="live"/);
  assert.match(html, /Predict the Swiss Stage/);
  assert.match(html, /term-advanced/);
  assert.match(html, /term-eliminated/);
  assert.match(html, /pickem-dock/);
  assert.match(html, /matchup-shell predict-shell view-simple/);

  assert.equal((html.match(/predict-card/g) || []).length, 33, "33 Swiss matches");
  assert.equal((html.match(/data-predict-mk=/g) || []).length, 66, "two pick buttons per match");
});

test("predict board shows the divergence note + extra SIM tags after an upset", () => {
  const picks = picksFromResults(stage.results);
  picks[matchKey("SINNERS", "FlyQuest")] = "SINNERS";
  const b = buildSwiss(picks, realData);
  const html = renderPredictHtml(b, null, "simple", { pickCount: countPicks(picks, b.validKeys) });
  assert.match(html, /已偏离真实赛果/);
  assert.match(html, /aria-pressed="true"/);
});

test("blank predict board renders only the 8 opening cards", () => {
  const b = buildSwiss({}, realData);
  const html = renderPredictHtml(b, null, "simple", { pickCount: 0 });
  assert.equal((html.match(/predict-card/g) || []).length, 8);
  assert.match(html, /与真实赛果一致/);
});

test("predict cards carry FLIP keys, actionable hints, and terminal trace hooks", () => {
  const blank = buildSwiss({}, realData);
  const blankHtml = renderPredictHtml(blank, null, "simple", { pickCount: 0 });
  assert.equal((blankHtml.match(/data-mk=/g) || []).length, 8, "one FLIP key per card");
  assert.equal((blankHtml.match(/is-actionable/g) || []).length, 8, "all opening matches actionable");

  const picks = picksFromResults(stage.results);
  const full = buildSwiss(picks, realData);
  const fullHtml = renderPredictHtml(full, null, "simple", { pickCount: countPicks(picks, full.validKeys) });
  assert.match(fullHtml, /term-chip/, "fully-played board has terminal chips");
  assert.ok(/term-chip[^>]*data-predict-team=/.test(fullHtml), "terminal chips carry trace hook");
});

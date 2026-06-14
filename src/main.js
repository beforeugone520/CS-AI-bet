import { loadSiteData } from "./data.js";
import { resetSwissState, applySwissWinner, clearSwissSelections, fixtureKey, groupSwissRecords, undoSwiss } from "./swiss.js";
import { emptyBracketState, applyBracketWinner } from "./bracket.js";
import { classifyPickem, summarizePickem } from "./pickem.js";
import { buildSwiss, picksFromResults, realRoundsFromStage, countPicks } from "./swissSim.js";
import { renderApp, renderHero } from "./render.js";
import { renderPredictWorkspace } from "./renderPredict.js";
import { initChrome, afterRender, setSignal, initPredictTrace, captureRects, playFlip, showConflictToast } from "./effects.js";
import { diffPrunedPicks } from "./trace.js";

const root = document.querySelector("#app");

let siteData = null;       // raw loaded payload
let appData = null;        // prepared (live mode)
let swissState = null;
let bracketState = null;
let swissViewMode = "simple";

// PREDICT mode state
let swissMode = "predict"; // 'predict' (majors.im sim) | 'live' (real results board)
let realData = null;       // { realByRound, realWinners }
let predictPicks = {};     // { matchKey: winner }
let predictOrder = [];     // pick order for undo

const handlers = {
  onSwissWinner, onSwissUndo, onSwissReset, onSwissViewMode, onBracketWinner,
  onSwissMode, onPredictPick, onPredictUndo, onPredictReset, onPredictLoadReal
};

initChrome();
loadCurrentRoute();
window.addEventListener("hashchange", loadCurrentRoute);

function loadCurrentRoute() {
  loadSiteData(window.location.hash || "#/")
    .then((data) => {
      siteData = data;
      swissState = null;
      bracketState = null;
      realData = null;
      predictPicks = {};
      predictOrder = [];
      if (data.stage.format === "swiss" && !data.stage.empty_state) {
        swissState = resetSwissState(data.stage.standings);
        realData = realRoundsFromStage(data.stage);
        predictPicks = picksFromResults(data.stage.results);
        predictOrder = Object.keys(predictPicks);
      }
      if (data.stage.format === "playoff" && !data.stage.empty_state) {
        bracketState = emptyBracketState(data.stage.bracket);
      }
      appData = prepareAppData(data);
      setSignal(data.sourceStatus, data.latest);
      paint(true);
    })
    .catch((error) => {
      root.innerHTML = `<section class="simulator-page"><div class="future-stage-card hud"><span class="hud-c1"></span><span class="hud-c2"></span><span class="future-stage-kicker">SIGNAL LOST</span><h2>数据加载失败</h2><p class="muted">${escapeText(error.message)}</p></div></section>`;
    });
}

function isPredictable() {
  return siteData && siteData.stage && siteData.stage.format === "swiss" && !siteData.stage.empty_state;
}

function paint(animate) {
  if (isPredictable() && swissMode === "predict") {
    renderPredictBoard(animate);
  } else {
    renderApp(root, appData, handlers);
    afterRender(root, { animate });
  }
}

/* ---------- PREDICT mode ---------- */
function renderPredictBoard(animate) {
  let bracket = buildSwiss(predictPicks, realData);
  // drop picks that no longer reference a generated matchup, or whose winner is
  // not actually a participant of that matchup, then rebuild
  const teamsByKey = new Map();
  for (const rd of bracket.rounds) for (const m of rd.matches) teamsByKey.set(m.key, [m.team1, m.team2]);
  const pruned = {};
  for (const [k, v] of Object.entries(predictPicks)) {
    const teams = teamsByKey.get(k);
    if (teams && teams.includes(v)) pruned[k] = v;
  }
  if (Object.keys(pruned).length !== Object.keys(predictPicks).length) {
    predictPicks = pruned;
    predictOrder = predictOrder.filter((k) => Object.prototype.hasOwnProperty.call(pruned, k));
    bracket = buildSwiss(predictPicks, realData);
  }

  const records = {};
  for (const s of bracket.standings) records[s.team] = s;
  const pickem = siteData.pickem;
  const rows = pickem && pickem.picks ? classifyPickem(pickem, records) : null;
  const pickemRuntime = rows ? { rows, summary: summarizePickem(rows) } : null;

  const heroData = {
    ...siteData,
    stage: { ...siteData.stage, standings: bracket.standings },
    pickemRuntime,
    swissViewMode
  };

  root.innerHTML = renderHero(heroData);
  const predictor = root.querySelector("#predictor");
  if (predictor) {
    renderPredictWorkspace(predictor, bracket, handlers, pickemRuntime, swissViewMode, {
      pickCount: countPicks(predictPicks, bracket.validKeys)
    });
  }
  afterRender(root, { animate });
  initPredictTrace(root, bracket);
}

function onSwissMode(mode) {
  swissMode = mode === "live" ? "live" : "predict";
  paint(false);
}

function onPredictPick(mk, team) {
  if (!mk || !team) return;
  const oldRects = captureRects(root);
  if (predictPicks[mk] === team) {
    delete predictPicks[mk];                         // click winner again -> deselect
    predictOrder = predictOrder.filter((k) => k !== mk);
  } else {
    predictPicks[mk] = team;
    if (!predictOrder.includes(mk)) predictOrder.push(mk);
  }
  // surface downstream picks this edit invalidates, before the board prunes them
  const dropped = diffPrunedPicks(predictPicks, buildSwiss(predictPicks, realData)).filter((d) => d.key !== mk);
  renderPredictBoard(false);
  playFlip(root, oldRects);
  if (dropped.length) showConflictToast(dropped);
}

function onPredictUndo() {
  const last = predictOrder.pop();
  if (last) delete predictPicks[last];
  renderPredictBoard(false);
}

function onPredictReset() {
  predictPicks = {};
  predictOrder = [];
  renderPredictBoard(false);
}

function onPredictLoadReal() {
  predictPicks = picksFromResults(siteData.stage.results);
  predictOrder = Object.keys(predictPicks);
  renderPredictBoard(false);
}

/* ---------- LIVE mode (existing) ---------- */
function onSwissWinner(fixtureIndex, winner) {
  const fixture = appData.stage.fixtures[fixtureIndex];
  swissState = applySwissWinner(swissState, fixture, winner);
  rerenderLive();
}

function onSwissUndo() {
  swissState = undoSwiss(swissState);
  rerenderLive();
}

function onSwissReset() {
  swissState = clearSwissSelections(swissState);
  rerenderLive();
}

function onSwissViewMode(mode) {
  swissViewMode = mode;
  paint(false);
}

function onBracketWinner(matchId, winner) {
  bracketState = applyBracketWinner(bracketState, matchId, winner);
  appData = enrichPickem({
    ...appData,
    stage: {
      ...appData.stage,
      bracket: bracketPayloadFromState(bracketState),
      champion_path: { champion: bracketState.champion }
    }
  });
  renderApp(root, appData, handlers);
  afterRender(root, { animate: false });
}

function rerenderLive() {
  appData = prepareAppData(appData);
  renderApp(root, appData, handlers);
  afterRender(root, { animate: false });
}

function prepareAppData(data) {
  if (!swissState || data.stage.empty_state || data.stage.format !== "swiss") {
    return enrichPickem({ ...data, swissViewMode });
  }
  const selectedByKey = {};
  for (const entry of swissState.history) {
    selectedByKey[entry.key || fixtureKey(entry.fixture)] = entry;
  }
  return enrichPickem({
    ...data,
    swissViewMode,
    stage: {
      ...data.stage,
      real_standings: data.stage.real_standings || data.stage.standings,
      standings: Object.values(swissState.records),
      simulation: {
        history: swissState.history,
        selected_by_key: selectedByKey,
        groups: groupSwissRecords(swissState.records)
      }
    }
  });
}

function enrichPickem(data) {
  if (!data.pickem || !data.pickem.picks || data.stage.empty_state || data.stage.format !== "swiss") {
    return data;
  }
  const records = {};
  for (const row of data.stage.standings) {
    records[row.team] = row;
  }
  const rows = classifyPickem(data.pickem, records);
  return {
    ...data,
    pickemRuntime: { rows, summary: summarizePickem(rows) }
  };
}

function bracketPayloadFromState(state) {
  const bracket = { quarterfinals: [], semifinals: [], final: [] };
  for (const match of Object.values(state.matches)) {
    if (!bracket[match.round]) bracket[match.round] = [];
    bracket[match.round].push(match);
  }
  return bracket;
}

function escapeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

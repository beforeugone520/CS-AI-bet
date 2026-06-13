import { loadSiteData } from "./data.js";
import { resetSwissState, applySwissWinner, clearSwissSelections, fixtureKey, groupSwissRecords, undoSwiss } from "./swiss.js";
import { emptyBracketState, applyBracketWinner } from "./bracket.js";
import { classifyPickem, summarizePickem } from "./pickem.js";
import { renderApp } from "./render.js";
import { initChrome, afterRender, setSignal } from "./effects.js";

const root = document.querySelector("#app");
let appData = null;
let swissState = null;
let bracketState = null;
let swissViewMode = "simple";

const handlers = { onSwissWinner, onSwissUndo, onSwissReset, onSwissViewMode, onBracketWinner };

initChrome();
loadCurrentRoute();
window.addEventListener("hashchange", loadCurrentRoute);

function loadCurrentRoute() {
  loadSiteData(window.location.hash || "#/")
    .then((data) => {
      swissState = null;
      bracketState = null;
      if (data.stage.format === "swiss" && !data.stage.empty_state) {
        swissState = resetSwissState(data.stage.standings);
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

function paint(animate) {
  renderApp(root, appData, handlers);
  afterRender(root, { animate });
}

function onSwissWinner(fixtureIndex, winner) {
  const fixture = appData.stage.fixtures[fixtureIndex];
  swissState = applySwissWinner(swissState, fixture, winner);
  rerender();
}

function onSwissUndo() {
  swissState = undoSwiss(swissState);
  rerender();
}

function onSwissReset() {
  swissState = clearSwissSelections(swissState);
  rerender();
}

function onSwissViewMode(mode) {
  swissViewMode = mode;
  rerender();
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
  paint(false);
}

function rerender() {
  appData = prepareAppData(appData);
  paint(false);
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

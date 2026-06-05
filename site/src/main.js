import { loadSiteData } from "./data.js";
import { resetSwissState, applySwissWinner } from "./swiss.js";
import { emptyBracketState, applyBracketWinner } from "./bracket.js";
import { classifyPickem, summarizePickem } from "./pickem.js";
import { renderApp } from "./render.js";

const root = document.querySelector("#app");
let appData = null;
let swissState = null;
let bracketState = null;

loadCurrentRoute();
window.addEventListener("hashchange", loadCurrentRoute);

function loadCurrentRoute() {
  loadSiteData(window.location.hash || "#/")
    .then((data) => {
      appData = enrichPickem(data);
      swissState = null;
      bracketState = null;
      if (data.stage.format === "swiss" && !data.stage.empty_state) {
        swissState = resetSwissState(data.stage.standings);
      }
      if (data.stage.format === "playoff" && !data.stage.empty_state) {
        bracketState = emptyBracketState(data.stage.bracket);
      }
      renderApp(root, appData, { onSwissWinner, onBracketWinner });
    })
    .catch((error) => {
      root.innerHTML = `<section class="panel panel-head"><h1>数据加载失败</h1><p class="muted">${error.message}</p></section>`;
    });
}

function onSwissWinner(fixtureIndex, winner) {
  const fixture = appData.stage.fixtures[fixtureIndex];
  swissState = applySwissWinner(swissState, fixture, winner);
  appData = enrichPickem({
    ...appData,
    stage: {
      ...appData.stage,
      standings: Object.values(swissState.records)
    }
  });
  renderApp(root, appData, { onSwissWinner, onBracketWinner });
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
  renderApp(root, appData, { onSwissWinner, onBracketWinner });
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
    pickemRuntime: {
      rows,
      summary: summarizePickem(rows)
    }
  };
}

function bracketPayloadFromState(state) {
  const bracket = { quarterfinals: [], semifinals: [], final: [] };
  for (const match of Object.values(state.matches)) {
    bracket[match.round].push(match);
  }
  return bracket;
}

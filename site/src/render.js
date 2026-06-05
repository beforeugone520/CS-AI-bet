import { fixtureKey } from "./swiss.js";

export function renderApp(root, data, handlers) {
  root.innerHTML = `
    <section class="simulator-page">
      <h1>IEM Cologne Major 2026 Simulator</h1>
      <div class="update-strip">
        <span>${escapeHtml(data.sourceStatus.visible_status || data.latest.source_status)}</span>
        <span class="mono">Updated ${escapeHtml(data.latest.last_updated)}</span>
      </div>
      <div id="predictor"></div>
    </section>
  `;
  renderPredictor(root.querySelector("#predictor"), data.stage, handlers, data.pickemRuntime, data.swissViewMode || "simple");
}

export function renderPredictor(root, stage, handlers, pickemRuntime = null, viewMode = "simple") {
  if (stage.empty_state) {
    root.innerHTML = renderFutureStage(stage, viewMode);
    return;
  }
  if (stage.format === "swiss") {
    root.innerHTML = renderSwissWorkspace(stage, pickemRuntime, viewMode);
    root.querySelectorAll("[data-winner]").forEach((button) => {
      button.addEventListener("click", () => handlers.onSwissWinner(Number(button.dataset.index), button.dataset.winner));
    });
    root.querySelectorAll("[data-view-mode]").forEach((button) => {
      button.addEventListener("click", () => handlers.onSwissViewMode?.(button.dataset.viewMode));
    });
    root.querySelector("[data-swiss-undo]")?.addEventListener("click", () => handlers.onSwissUndo());
    root.querySelector("[data-swiss-reset]")?.addEventListener("click", () => handlers.onSwissReset());
    return;
  }
  if (stage.format === "playoff") {
    const bracket = stage.bracket || {};
    root.innerHTML = `
      <div class="panel-head"><h2>Playoff Bracket</h2><span class="mono muted">${escapeHtml(stage.stage_id)}</span></div>
      ${renderBracketRound("Quarterfinals", bracket.quarterfinals || [])}
      ${renderBracketRound("Semifinals", bracket.semifinals || [])}
      ${renderBracketRound("Final", bracket.final || [])}
    `;
    root.querySelectorAll("[data-bracket-winner]").forEach((button) => {
      button.addEventListener("click", () => handlers.onBracketWinner(button.dataset.matchId, button.dataset.bracketWinner));
    });
    return;
  }
  root.innerHTML = `<div class="panel-head"><h2>Stage data unavailable</h2><p class="muted">This stage format is not supported yet.</p></div>`;
}

export function renderStatusBar(data) {
  return `
    <section class="panel panel-head" aria-label="数据状态">
      <div>
        <strong>${escapeHtml(data.latest.event_id)}</strong>
        <div class="muted">Last updated: ${escapeHtml(data.latest.last_updated)}</div>
      </div>
      <div class="mono ${sourceStatusClass(data.latest.source_status)}">${escapeHtml(data.sourceStatus.visible_status || data.latest.source_status)}</div>
    </section>
  `;
}

function renderStageHead(stage) {
  return `<div class="panel-head"><div><h1>${escapeHtml(stage.name || stage.stage_id)}</h1><p class="muted">${escapeHtml(stage.format)} · ${escapeHtml(stage.status)}</p></div></div>`;
}

function renderSwissWorkspace(stage, runtime, viewMode = "simple") {
  const simulation = stage.simulation || { history: [], selected_by_key: {}, groups: null };
  const groups = simulation.groups || groupFromRows(stage.standings || []);
  const fixtureIndexByKey = fixtureIndexes(stage.fixtures || []);
  const boardRounds = roundsForBoard(stage);
  return `
    <div class="matchup-shell view-${escapeHtml(viewMode)}">
      ${renderStageControls(stage.stage_id, viewMode)}
      <div class="matchup-header ${simulation.history.length ? "has-local-picks" : ""}">
        <div>
          <h2>Stage 1 Swiss Matchups</h2>
          <p>Locked results and local Round 5 picks.</p>
        </div>
        <div class="matchup-controls" aria-label="Swiss controls">
          <span class="mono">${simulation.history.length} picks</span>
          <button class="winner-button compact-button" data-swiss-undo ${simulation.history.length ? "" : "disabled"}>Undo</button>
          <button class="winner-button compact-button" data-swiss-reset ${simulation.history.length ? "" : "disabled"}>Reset</button>
        </div>
      </div>
      <div class="matchup-summary">
        <span><strong>${stage.fixtures.length}</strong> remaining BO3</span>
        <span><strong>${groups.advanced.length}</strong> advanced</span>
        <span><strong>${groups.live.length}</strong> live</span>
        <span><strong>${groups.eliminated.length}</strong> eliminated</span>
      </div>
      <section class="swiss-round-board" aria-label="Swiss round board">
        ${boardRounds.map((round, index) => `
          ${renderRoundColumn(round, simulation.selected_by_key, fixtureIndexByKey)}
          ${index < boardRounds.length - 1 ? renderRoundFlowArrow(round.round, boardRounds[index + 1].round) : ""}
        `).join("")}
      </section>
      ${renderPickemImpact(runtime)}
      <section class="standings-board">
        ${renderRecordGroup("Advanced", groups.advanced, "status-good")}
        ${renderRecordGroup("Live", groups.live, "status-warn")}
        ${renderRecordGroup("Eliminated", groups.eliminated, "status-bad")}
      </section>
    </div>
  `;
}

function renderFutureStage(stage, viewMode = "simple") {
  return `
    <div class="matchup-shell">
      ${renderStageControls(stage.stage_id, viewMode)}
      <section class="future-stage-page" aria-label="${escapeHtml(stage.stage_id)} status">
        <div class="future-stage-card">
          <span class="future-stage-kicker">${escapeHtml(stage.stage_id)} · ${escapeHtml(stage.format)} · ${escapeHtml(stage.status)}</span>
          <h2>${escapeHtml(stage.empty_state.title)}</h2>
          <p>${escapeHtml(stage.empty_state.message)}</p>
          <div class="future-stage-meta">
            <span>Only IEM Cologne Major 2026 data is shown here.</span>
            <span class="mono">${escapeHtml(stage.empty_state.next_update)}</span>
          </div>
        </div>
      </section>
    </div>
  `;
}

function renderStageControls(currentStageId, viewMode = "simple") {
  const current = currentStageId || "stage-1";
  const mode = normalizeViewMode(viewMode);
  return `
    <div class="stage-control-row">
      <div class="stage-tabs" aria-label="Stage controls">
        <a class="stage-tab ${current === "stage-1" ? "active" : ""}" href="#/stage/1" aria-label="Stage 1"><span class="stage-label-full">Stage 1</span><span class="stage-label-short">S1</span></a>
        <a class="stage-tab ${current === "stage-2" ? "active" : ""}" href="#/stage/2" aria-label="Stage 2"><span class="stage-label-full">Stage 2</span><span class="stage-label-short">S2</span></a>
        <a class="advance-button ${current === "stage-3" ? "active" : ""}" href="#/stage/3" aria-label="Advance"><span class="stage-label-full">Advance →</span><span class="stage-label-short">→</span></a>
      </div>
      <div class="view-switcher" aria-label="View switcher">
        <button class="${mode === "simple" ? "active" : ""}" type="button" title="Simple View" aria-label="Simple View" data-view-mode="simple">♕</button>
        <button class="${mode === "minimal" ? "active" : ""}" type="button" title="Minimal View" aria-label="Minimal View" data-view-mode="minimal">☰</button>
        <button class="${mode === "bracket" ? "active" : ""}" type="button" title="Bracket View" aria-label="Bracket View" data-view-mode="bracket">▦</button>
        <button class="${mode === "classic" ? "active" : ""}" type="button" title="Classic View" aria-label="Classic View" data-view-mode="classic">▤</button>
      </div>
    </div>
  `;
}

function normalizeViewMode(mode) {
  return ["simple", "minimal", "bracket", "classic"].includes(mode) ? mode : "simple";
}

function renderRoundColumn(round, selectedByKey, fixtureIndexByKey) {
  const lockedMatches = (round.results || []).map((match) => renderSwissMatchCard(match, { locked: true }));
  const fixtureMatches = (round.fixtures || []).map((fixture) => {
    const key = fixtureKey(fixture);
    return renderSwissMatchCard(fixture, {
      locked: false,
      selection: selectedByKey[key],
      index: fixtureIndexByKey[key]
    });
  });
  const matches = lockedMatches.concat(fixtureMatches);
  return `
    <section class="round-column">
      <div class="round-heading">
        <h3>Round ${escapeHtml(round.round)}</h3>
        <span>${roundLabel(round)}</span>
      </div>
      <div class="round-stack">
        ${matches.length ? matches.join("") : `<div class="empty-round">Waiting for real data</div>`}
      </div>
    </section>
  `;
}

function renderRoundFlowArrow(fromRound, toRound) {
  return `
    <div class="round-flow-arrow" aria-hidden="true" title="Round ${escapeHtml(fromRound)} to Round ${escapeHtml(toRound)}">
      <span>→</span>
    </div>
  `;
}

function renderSwissMatchCard(match, options) {
  const selection = options.selection || null;
  const selectedWinner = selection?.winner || (options.locked ? match.winner : null);
  const team1Selected = selectedWinner === match.team1;
  const team2Selected = selectedWinner === match.team2;
  const className = options.locked ? "locked-match" : "fixture-match";
  const status = matchStatus(match, options, selectedWinner);
  return `
    <div class="swiss-match-card ${className} ${selectedWinner ? "has-winner" : ""} ${status.className}">
      ${status.label ? `<div class="match-ribbon"><span>${escapeHtml(status.label)}</span><strong>${escapeHtml(status.record)}</strong></div>` : ""}
      <div class="match-card-body">
        ${renderSwissTeamButton(match.team1, team1Selected, options, match, "left")}
        <div class="match-score">${escapeHtml(displayScore(match, options, selectedWinner))}</div>
        ${renderSwissTeamButton(match.team2, team2Selected, options, match, "right")}
      </div>
      <div class="match-card-meta">
        <span>${escapeHtml(options.locked ? "locked" : "pick")}</span>
        <span class="mono">${escapeHtml(matchMeta(match, options))}</span>
      </div>
    </div>
  `;
}

function renderSwissTeamButton(team, selected, options, match, side) {
  const logoFile = teamLogoFile(team);
  const logo = logoFile
    ? `<img class="team-logo" src="./assets/teams/${escapeHtml(logoFile)}.png" alt="" loading="lazy" decoding="async">`
    : "";
  const body = `<span class="team-mark team-${escapeHtml(teamSlug(team))}" aria-hidden="true">${logo}<span class="team-fallback">${escapeHtml(teamMark(team))}</span></span><span class="team-name">${escapeHtml(team)}</span>`;
  if (options.locked || options.index === undefined) {
    return `<div class="team-slot ${side}-slot ${selected ? "winner-slot" : "loser-slot"}">${body}</div>`;
  }
  return `
    <button class="team-slot ${side}-slot pick-slot ${selected ? "winner-slot" : ""}" data-index="${options.index}" data-winner="${escapeHtml(team)}">${body}</button>
  `;
}

function renderMatchRow(fixture, index, selection = null) {
  const selectedWinner = selection?.winner;
  const team1Selected = selectedWinner === fixture.team1;
  const team2Selected = selectedWinner === fixture.team2;
  return `
    <div class="match-row ${selectedWinner ? "selected-row" : ""}">
      <div>
        <strong>${escapeHtml(fixture.team1)} vs ${escapeHtml(fixture.team2)}</strong>
        <div class="muted">${escapeHtml(fixture.note || fixture.swiss_match_type || "")}</div>
      </div>
      <div class="match-actions">
        <button class="winner-button ${team1Selected ? "is-selected" : ""}" data-index="${index}" data-winner="${escapeHtml(fixture.team1)}">${escapeHtml(fixture.team1)}</button>
        <button class="winner-button ${team2Selected ? "is-selected" : ""}" data-index="${index}" data-winner="${escapeHtml(fixture.team2)}">${escapeHtml(fixture.team2)}</button>
      </div>
    </div>
  `;
}

function renderSimulationHistory(history) {
  return `
    <section class="swiss-section">
      <div class="section-label"><h3>Local Simulation</h3><span class="muted">${history.length ? "Selections are local only." : "No simulated winners selected."}</span></div>
      ${history.length ? history.map((entry, index) => `
        <div class="team-row">
          <span>${index + 1}. ${escapeHtml(entry.fixture.team1)} vs ${escapeHtml(entry.fixture.team2)}</span>
          <span class="mono status-pill">${escapeHtml(entry.winner)} wins</span>
        </div>
      `).join("") : `<div class="team-row"><span class="muted">Pick winners in the fixture list to preview final records.</span></div>`}
    </section>
  `;
}

function renderPickemImpact(runtime) {
  if (!runtime) {
    return `
      <section class="pickem-dock">
        <div class="section-label"><h3>Pick'em Impact</h3><span class="muted">Waiting for pick data.</span></div>
      </section>
    `;
  }
  return `
    <section class="pickem-dock">
      <div class="section-label">
        <h3>Pick'em Impact</h3>
        <span class="muted">${runtime.summary.locked} locked / ${runtime.summary.alive} alive / ${runtime.summary.broken} broken / ${runtime.summary.missing} missing</span>
      </div>
      <div class="pickem-slots">
        ${runtime.rows.map((row) => `
          <div class="pickem-chip ${statusClass(row.status)}">
            <strong>${escapeHtml(row.team)}</strong>
            <span>${escapeHtml(row.category)} · ${row.wins ?? "-"}-${row.losses ?? "-"} · ${escapeHtml(row.status)}</span>
          </div>
        `).join("")}
      </div>
    </section>
  `;
}

function renderRecordGroup(label, rows, className) {
  return `
    <section class="record-group">
      <div class="section-label"><h3>${escapeHtml(label)}</h3><span class="${className} mono">${rows.length}</span></div>
      ${rows.length ? rows.map(renderStandingRow).join("") : `<div class="team-row"><span class="muted">None</span></div>`}
    </section>
  `;
}

function renderBracketRound(label, matches) {
  return `
    <section class="bracket-round">
      <div class="panel-head"><h3>${escapeHtml(label)}</h3></div>
      ${matches.length ? matches.map(renderBracketMatch).join("") : `<div class="match-row"><span class="muted">Waiting for bracket draw.</span></div>`}
    </section>
  `;
}

function renderBracketMatch(match) {
  const team1 = match.team1 || "待定";
  const team2 = match.team2 || "待定";
  return `
    <div class="match-row">
      <div><strong>${escapeHtml(team1)} vs ${escapeHtml(team2)}</strong><div class="muted">${escapeHtml(match.winner ? `Winner: ${match.winner}` : match.id)}</div></div>
      <div class="match-actions">
        ${match.team1 ? `<button class="winner-button" data-match-id="${escapeHtml(match.id)}" data-bracket-winner="${escapeHtml(match.team1)}">${escapeHtml(match.team1)}</button>` : ""}
        ${match.team2 ? `<button class="winner-button" data-match-id="${escapeHtml(match.id)}" data-bracket-winner="${escapeHtml(match.team2)}">${escapeHtml(match.team2)}</button>` : ""}
      </div>
    </div>
  `;
}

function renderStandingRow(row) {
  return `<div class="team-row"><span>${escapeHtml(row.team)}</span><span class="mono status-pill">${row.wins}-${row.losses} ${escapeHtml(row.status)}</span></div>`;
}

function groupFromRows(rows) {
  return {
    advanced: rows.filter((row) => row.status === "advanced"),
    live: rows.filter((row) => row.status === "alive"),
    eliminated: rows.filter((row) => row.status === "eliminated")
  };
}

function roundsForBoard(stage) {
  const byRound = new Map();
  for (const round of stage.rounds || []) {
    byRound.set(String(round.round), {
      round: String(round.round),
      results: round.results || [],
      fixtures: round.fixtures || []
    });
  }
  for (const result of stage.results || []) {
    const key = String(result.round || "unknown");
    if (!byRound.has(key)) byRound.set(key, { round: key, results: [], fixtures: [] });
    const exists = byRound.get(key).results.some((item) => resultKey(item) === resultKey(result));
    if (!exists) byRound.get(key).results.push(result);
  }
  for (const fixture of stage.fixtures || []) {
    const key = String(fixture.swiss_round || fixture.round || "next");
    if (!byRound.has(key)) byRound.set(key, { round: key, results: [], fixtures: [] });
    const exists = byRound.get(key).fixtures.some((item) => fixtureKey(item) === fixtureKey(fixture));
    if (!exists) byRound.get(key).fixtures.push(fixture);
  }
  return Array.from(byRound.values()).sort((a, b) => Number(a.round) - Number(b.round));
}

function fixtureIndexes(fixtures) {
  const indexes = {};
  fixtures.forEach((fixture, index) => {
    indexes[fixtureKey(fixture)] = index;
  });
  return indexes;
}

function roundLabel(round) {
  const number = String(round.round);
  const labels = {
    "1": "0-0",
    "2": "1-0 / 0-1",
    "3": "2-0 / 1-1 / 0-2",
    "4": "2-1 / 1-2",
    "5": "2-2 deciders"
  };
  return labels[number] || "Swiss";
}

function displayScore(match, options, selectedWinner) {
  if (!options.locked) {
    return selectedWinner ? "3 : 2" : "vs";
  }
  const mapScores = String(match.map_scores || "");
  const score = mapScores && !mapScores.includes(";") ? mapScores : String(match.match_score || "");
  return score.replace(/\s*-\s*/g, " : ");
}

function matchMeta(match, options) {
  if (options.locked) {
    return String(match.note || match.maps || "");
  }
  return [match.team1_record, match.team2_record].filter(Boolean).join(" / ");
}

function matchStatus(match, options, selectedWinner) {
  const text = String(match.note || match.swiss_match_type || "").toLowerCase();
  if (!options.locked && String(match.swiss_round || match.round) === "5") {
    return {
      className: selectedWinner ? "advance-card" : "",
      label: selectedWinner ? "ADVANCE" : "",
      record: selectedWinner ? "3:2" : ""
    };
  }
  if (text.includes("elimination")) {
    return { className: "elimination-card", label: "ELIMINATED", record: "1:3" };
  }
  if (text.includes("advancement")) {
    return { className: "advance-card", label: "ADVANCE", record: "3:1" };
  }
  return { className: "", label: "", record: "" };
}

function teamInitials(team) {
  return String(team || "?")
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part[0])
    .join("")
    .slice(0, 3)
    .toUpperCase();
}

function teamMark(team) {
  const marks = {
    "BetBoom": "BB",
    "FlyQuest": "FQ",
    "Gaimin Gladiators": "GG",
    "GamerLegion": "GL",
    "HEROIC": "H",
    "Liquid": "TL",
    "Lynn Vision": "LV",
    "Sharks": "SH",
    "SINNERS": "SIN",
    "THUNDER dOWNUNDER": "TD",
    "TYLOO": "TY",
  };
  return marks[team] || teamInitials(team);
}

function teamLogoFile(team) {
  const logos = {
    "B8": "b8",
    "BIG": "big",
    "BetBoom": "betb",
    "FlyQuest": "fly",
    "Gaimin Gladiators": "gg",
    "GamerLegion": "gl",
    "HEROIC": "hero",
    "Liquid": "liqu",
    "Lynn Vision": "lvg",
    "M80": "m80",
    "MIBR": "mibr",
    "NRG": "nrg",
    "SINNERS": "sinn",
    "Sharks": "shks",
    "THUNDER dOWNUNDER": "tdu",
    "TYLOO": "tylo",
  };
  return logos[team] || "";
}

function teamSlug(team) {
  return String(team).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "team";
}

function resultKey(result) {
  return [
    result.round || "round",
    result.team1 || "",
    result.team2 || "",
    result.winner || "",
    result.match_score || ""
  ].join(":");
}

function sourceStatusClass(status) {
  if (status === "cached" || status === "fallback_success") return "status-warn";
  if (status === "failed") return "status-bad";
  return "status-good";
}

function statusClass(status) {
  if (status === "locked") return "pickem-locked";
  if (status === "broken") return "pickem-broken";
  if (status === "alive") return "pickem-alive";
  return "";
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}

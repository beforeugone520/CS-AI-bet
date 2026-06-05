import { fixtureKey } from "./swiss.js";

export function renderApp(root, data, handlers) {
  if (data.route === "#/ai") {
    root.innerHTML = `${renderStatusBar(data)}${renderAiArchive(data)}`;
    return;
  }
  if (data.route === "#/model") {
    root.innerHTML = `${renderStatusBar(data)}${renderModelLab(data)}`;
    return;
  }
  root.innerHTML = `
    ${renderStatusBar(data)}
    <section class="command-grid">
      <div class="panel">
        ${renderStageHead(data.stage)}
        <div id="predictor"></div>
      </div>
      <aside class="panel">
        ${renderAiDesk(data)}
      </aside>
    </section>
  `;
  renderPredictor(root.querySelector("#predictor"), data.stage, handlers, data.pickemRuntime);
}

export function renderPredictor(root, stage, handlers, pickemRuntime = null) {
  if (stage.empty_state) {
    root.innerHTML = `
      <div class="panel-head">
        <div>
          <h2>${escapeHtml(stage.empty_state.title)}</h2>
          <p class="muted">${escapeHtml(stage.empty_state.message)}</p>
        </div>
        <span class="mono muted">${escapeHtml(stage.empty_state.next_update)}</span>
      </div>
    `;
    return;
  }
  if (stage.format === "swiss") {
    root.innerHTML = renderSwissWorkspace(stage, pickemRuntime);
    root.querySelectorAll("[data-winner]").forEach((button) => {
      button.addEventListener("click", () => handlers.onSwissWinner(Number(button.dataset.index), button.dataset.winner));
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

export function renderAiDesk(data) {
  const articles = data.articles.articles || [];
  const runtime = data.pickemRuntime;
  const fallback = Boolean(data.articles.fallback_used);
  const runtimeHtml = runtime ? `
    <div class="article-row">
      <div>
        <strong>Pick'em runtime</strong>
        <p class="muted">${runtime.summary.locked} locked / ${runtime.summary.alive} alive / ${runtime.summary.broken} broken / ${runtime.summary.missing} missing</p>
      </div>
      <span class="mono muted">local</span>
    </div>
  ` : "";
  return `
    <div class="panel-head"><h2>AI Desk</h2><span class="mono muted">${fallback ? "template fallback" : `${articles.length} articles`}</span></div>
    ${fallback ? `<div class="truth-note">AI API 未生成正式内容；以下是基于静态数据的模板提示，不作为新闻稿。</div>` : ""}
    ${runtimeHtml}
    ${articles.map((article) => `
      <article class="article-row">
        <div>
          <strong>${escapeHtml(article.title)}</strong>
          <p class="muted">${escapeHtml(article.summary)}</p>
        </div>
        <span class="mono muted">${escapeHtml(article.type)}</span>
      </article>
    `).join("")}
  `;
}

function renderAiArchive(data) {
  const articles = data.articles.articles || [];
  const fallback = Boolean(data.articles.fallback_used);
  return `
    <section class="panel page-grid">
      <div class="panel-head">
        <div><h1>AI Desk</h1><p class="muted">${fallback ? "Template fallback from static data. Waiting for API-generated analysis." : "Generated analysis from static site data."}</p></div>
        <span class="mono muted">${fallback ? "template fallback" : "generated"}</span>
      </div>
      ${fallback ? `<div class="truth-note">当前内容不是实时新闻，也不是模型正式输出；它只用于说明 Pick'em 状态。</div>` : ""}
      ${articles.map((article) => `
        <article>
          <div class="article-row">
            <div><strong>${escapeHtml(article.title)}</strong><p class="muted">${escapeHtml(article.summary)}</p></div>
            <span class="mono muted">${escapeHtml(article.stage)} / ${escapeHtml(article.type)}</span>
          </div>
          <div class="article-body">${escapeHtml(article.body)}</div>
        </article>
      `).join("")}
    </section>
  `;
}

function renderModelLab(data) {
  const fallback = data.sourceStatus.fallback || {};
  return `
    <section class="panel">
      <div class="panel-head">
        <div><h1>Model Lab</h1><p class="muted">Static deployment, update schedule, source status, and forecast limits.</p></div>
        <span class="mono muted">02:00 BJT</span>
      </div>
      <div class="metric-row">
        <div class="metric"><strong>${escapeHtml(data.latest.current_stage)}</strong><span class="muted">current stage</span></div>
        <div class="metric"><strong>${escapeHtml(data.latest.source_status)}</strong><span class="muted">source status</span></div>
        <div class="metric"><strong>${escapeHtml(fallback.status || "unknown")}</strong><span class="muted">5E fallback</span></div>
        <div class="metric"><strong>${escapeHtml(data.articles.fallback_used ? "template" : "api")}</strong><span class="muted">AI mode</span></div>
      </div>
      <div class="article-body">Frontend reads static JSON only. GitHub Actions refreshes source data at 02:00 BJT, generates API articles when a secret is configured, and publishes the gh-pages branch. Unknown future stages stay locked until real data exists.</div>
    </section>
  `;
}

function renderStageHead(stage) {
  return `<div class="panel-head"><div><h1>${escapeHtml(stage.name || stage.stage_id)}</h1><p class="muted">${escapeHtml(stage.format)} · ${escapeHtml(stage.status)}</p></div></div>`;
}

function renderSwissWorkspace(stage, runtime) {
  const simulation = stage.simulation || { history: [], selected_by_key: {}, groups: null };
  const groups = simulation.groups || groupFromRows(stage.standings || []);
  return `
    <div class="panel-head swiss-title">
      <div>
        <h2>Swiss Matchup Simulator</h2>
        <p class="muted">真实进度锁定到当前 standings；下面只模拟剩余 fixtures，不会写回数据源。</p>
      </div>
      <span class="mono muted">${escapeHtml(stage.stage_id)} · ${simulation.history.length} local picks</span>
    </div>
    <div class="simulator-toolbar">
      <div class="metric compact"><strong>${stage.fixtures.length}</strong><span class="muted">remaining BO3</span></div>
      <div class="metric compact"><strong>${groups.advanced.length}</strong><span class="muted">advanced</span></div>
      <div class="metric compact"><strong>${groups.live.length}</strong><span class="muted">still live</span></div>
      <div class="metric compact"><strong>${groups.eliminated.length}</strong><span class="muted">eliminated</span></div>
      <div class="sim-actions">
        <button class="winner-button" data-swiss-undo ${simulation.history.length ? "" : "disabled"}>Undo</button>
        <button class="winner-button" data-swiss-reset ${simulation.history.length ? "" : "disabled"}>Reset</button>
      </div>
    </div>
    <section class="swiss-section">
      <div class="section-label"><h3>Round ${escapeHtml(stage.fixtures[0]?.swiss_round || "next")} Fixtures</h3><span class="muted">Click a winner to simulate.</span></div>
      ${stage.fixtures.map((fixture, index) => renderMatchRow(fixture, index, simulation.selected_by_key[fixtureKey(fixture)])).join("")}
    </section>
    ${renderSimulationHistory(simulation.history)}
    ${renderPickemImpact(runtime)}
    <section class="standings-board">
      ${renderRecordGroup("Advanced", groups.advanced, "status-good")}
      ${renderRecordGroup("Live", groups.live, "status-warn")}
      ${renderRecordGroup("Eliminated", groups.eliminated, "status-bad")}
    </section>
    ${renderLockedResults(stage.results || [])}
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
  if (!runtime) return "";
  return `
    <section class="swiss-section">
      <div class="section-label">
        <h3>Pick'em Impact</h3>
        <span class="muted">${runtime.summary.locked} locked / ${runtime.summary.alive} alive / ${runtime.summary.broken} broken / ${runtime.summary.missing} missing</span>
      </div>
      <div class="pickem-grid">
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

function renderLockedResults(results) {
  const latest = results.slice(-8).reverse();
  return `
    <section class="swiss-section locked-results">
      <div class="section-label"><h3>Locked Real Results</h3><span class="muted">${results.length} completed matches from static data</span></div>
      ${latest.map((row) => `
        <div class="team-row">
          <span>${escapeHtml(row.team1)} vs ${escapeHtml(row.team2)}</span>
          <span class="mono status-pill">${escapeHtml(row.winner)} ${escapeHtml(row.match_score || "")}</span>
        </div>
      `).join("")}
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

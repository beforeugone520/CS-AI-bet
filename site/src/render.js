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
  renderPredictor(root.querySelector("#predictor"), data.stage, handlers);
}

export function renderPredictor(root, stage, handlers) {
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
    root.innerHTML = `
      <div class="panel-head"><h2>Swiss Predictor</h2><span class="mono muted">${escapeHtml(stage.stage_id)}</span></div>
      ${stage.fixtures.map((fixture, index) => renderMatchRow(fixture, index)).join("")}
      <div id="standings">${stage.standings.map(renderStandingRow).join("")}</div>
    `;
    root.querySelectorAll("[data-winner]").forEach((button) => {
      button.addEventListener("click", () => handlers.onSwissWinner(Number(button.dataset.index), button.dataset.winner));
    });
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
    <div class="panel-head"><h2>AI Desk</h2><span class="mono muted">${articles.length} articles</span></div>
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
  return `
    <section class="panel page-grid">
      <div class="panel-head">
        <div><h1>AI Desk</h1><p class="muted">Generated analysis from static site data.</p></div>
        <span class="mono muted">${data.articles.fallback_used ? "fallback" : "generated"}</span>
      </div>
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
      <div class="article-body">Frontend reads static JSON only. GitHub Actions refreshes source data, generates articles, and deploys the Pages artifact.</div>
    </section>
  `;
}

function renderStageHead(stage) {
  return `<div class="panel-head"><div><h1>${escapeHtml(stage.name || stage.stage_id)}</h1><p class="muted">${escapeHtml(stage.format)} · ${escapeHtml(stage.status)}</p></div></div>`;
}

function renderMatchRow(fixture, index) {
  return `
    <div class="match-row">
      <div><strong>${escapeHtml(fixture.team1)} vs ${escapeHtml(fixture.team2)}</strong><div class="muted">${escapeHtml(fixture.note || fixture.swiss_match_type || "")}</div></div>
      <div class="match-actions">
        <button class="winner-button" data-index="${index}" data-winner="${escapeHtml(fixture.team1)}">${escapeHtml(fixture.team1)}</button>
        <button class="winner-button" data-index="${index}" data-winner="${escapeHtml(fixture.team2)}">${escapeHtml(fixture.team2)}</button>
      </div>
    </div>
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

function sourceStatusClass(status) {
  if (status === "cached" || status === "fallback_success") return "status-warn";
  if (status === "failed") return "status-bad";
  return "status-good";
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

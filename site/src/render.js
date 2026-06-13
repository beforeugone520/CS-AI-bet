import { fixtureKey } from "./swiss.js";

/* ============================================================================
   Tactical Intelligence Terminal — view layer
   Pure string rendering (DOM-safe for unit tests) + post-render event wiring.
   ========================================================================== */

export function renderHero(data) {
  const m = heroMetrics(data);
  const ss = data.sourceStatus || {};
  const latest = data.latest || {};
  const statusCls = sourceStatusClass(latest.source_status);
  return `
    <section class="simulator-page">
      <header class="command-hero hud reveal" style="--i:0">
        <span class="hud-c1"></span><span class="hud-c2"></span>
        <div class="command-hero__intel">
          <span class="kicker">// IEM COLOGNE MAJOR 2026 · STAGE 01 SWISS</span>
          <h1 class="command-hero__title">IEM Cologne Major 2026 Simulator</h1>
          <p class="command-hero__sub">完整可交互的瑞士轮推演 —— 逐场点选胜者、战绩池按种子 / Buchholz 自动重排，实时联动晋级名额与 <b>模型 / 专家 / 市场</b> 三路融合的 Pick'em 兑现。</p>
          <div class="update-strip">
            <span class="signal-chip ${statusCls}"><span class="dot"></span>${escapeHtml(ss.visible_status || latest.source_status || "—")}</span>
            <span class="mono">UPDATED · ${escapeHtml(formatStamp(latest.last_updated))}</span>
            <span class="mono">BUILD ${escapeHtml(latest.data_version || "—")}</span>
          </div>
        </div>
        <div class="command-hero__readout">
          <div class="stat-grid" aria-label="赛况概览">
            ${statCell("teams", "TEAMS · 战队", m.teams, "16 SEEDS")}
            ${statCell("secured", "SECURED · 晋级", m.advanced, "→ STAGE 2")}
            ${statCell("live", "IN PLAY · 存活", m.live, "R5 DECIDER")}
            ${statCell("out", "OUT · 淘汰", m.eliminated, "STAGE 1 END")}
            ${statPickem(m)}
          </div>
        </div>
      </header>
      <div id="predictor"></div>
    </section>
  `;
}

export function renderApp(root, data, handlers) {
  root.innerHTML = renderHero(data);
  const predictor = root.querySelector("#predictor");
  if (predictor) {
    renderPredictor(predictor, data.stage || {}, handlers, data.pickemRuntime, data.swissViewMode || "simple");
  }
}

export function renderModeSwitch(mode) {
  const m = mode === "live" ? "live" : "predict";
  return `
    <div class="mode-switch" role="group" aria-label="瑞士轮模式">
      <button class="mode-tab ${m === "predict" ? "active" : ""}" type="button" aria-pressed="${m === "predict"}" data-swiss-mode="predict">🎯 预测 PREDICT</button>
      <button class="mode-tab ${m === "live" ? "active" : ""}" type="button" aria-pressed="${m === "live"}" data-swiss-mode="live">🛰 实况 LIVE</button>
    </div>
  `;
}

function statCell(variant, label, value, sub) {
  const isNum = typeof value === "number" && Number.isFinite(value);
  return `
    <div class="stat stat--${variant}">
      <span class="stat__label">${escapeHtml(label)}</span>
      <strong class="stat__value mono"${isNum ? ` data-count="${value}"` : ""}>${escapeHtml(String(value))}</strong>
      <span class="stat__sub">${escapeHtml(sub)}</span>
    </div>
  `;
}

function statPickem(m) {
  return `
    <div class="stat stat--pickem" style="grid-column:1/-1">
      <span class="stat__label">PICK'EM · 提交答案兑现</span>
      <strong class="stat__value mono"><span data-count="${m.pickemLocked}">${m.pickemLocked}</span><span class="muted" style="font-size:.52em;letter-spacing:.04em"> / ${m.pickemTotal} SECURED</span></strong>
      <span class="stat__sub">ALIVE ${m.pickemAlive} · BROKEN ${m.pickemBroken}</span>
    </div>
  `;
}

function heroMetrics(data) {
  const stage = data.stage || {};
  const empty = !!stage.empty_state;
  const standings = stage.standings || [];
  const count = (s) => standings.filter((r) => r.status === s).length;
  const teams = (stage.teams && stage.teams.length) || standings.length || 16;
  const sum = (data.pickemRuntime && data.pickemRuntime.summary) ||
    (data.pickem && data.pickem.summary) || { locked: 0, alive: 0, broken: 0, missing: 0 };
  const total = (sum.locked || 0) + (sum.alive || 0) + (sum.broken || 0) + (sum.missing || 0);
  return {
    teams,
    advanced: empty ? "—" : count("advanced"),
    live: empty ? "—" : count("alive"),
    eliminated: empty ? "—" : count("eliminated"),
    pickemLocked: sum.locked || 0,
    pickemAlive: sum.alive || 0,
    pickemBroken: sum.broken || 0,
    pickemTotal: total || 10
  };
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
    root.querySelectorAll("[data-swiss-mode]").forEach((button) => {
      button.addEventListener("click", () => handlers.onSwissMode?.(button.dataset.swissMode));
    });
    root.querySelector("[data-swiss-undo]")?.addEventListener("click", () => handlers.onSwissUndo());
    root.querySelector("[data-swiss-reset]")?.addEventListener("click", () => handlers.onSwissReset());
    return;
  }
  if (stage.format === "playoff") {
    const bracket = stage.bracket || {};
    root.innerHTML = `
      <div class="matchup-shell view-${escapeHtml(viewMode)}">
        ${renderStageControls(stage.stage_id, viewMode)}
        <section class="pickem-dock">
          <div class="panel-head"><h2>Playoff Bracket</h2><span class="mono muted">${escapeHtml(stage.stage_id)}</span></div>
          ${renderBracketRound("Quarterfinals", bracket.quarterfinals || [])}
          ${renderBracketRound("Semifinals", bracket.semifinals || [])}
          ${renderBracketRound("Final", bracket.final || [])}
        </section>
      </div>
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

function renderSwissWorkspace(stage, runtime, viewMode = "simple") {
  const simulation = stage.simulation || { history: [], selected_by_key: {}, groups: null };
  const groups = simulation.groups || groupFromRows(stage.standings || []);
  const fixtureIndexByKey = fixtureIndexes(stage.fixtures || []);
  const boardRounds = roundsForBoard(stage);
  const picks = simulation.history.length;

  return `
    <div class="matchup-shell view-${escapeHtml(viewMode)}">
      ${renderStageControls(stage.stage_id, viewMode)}
      ${renderModeSwitch("live")}
      <div class="matchup-header ${picks ? "has-local-picks" : ""}">
        <div class="matchup-header__title">
          <span class="kicker">// SWISS BRACKET · 战绩池推进</span>
          <h2>Stage 1 Swiss Matchups</h2>
          <p>已锁定真实赛果 + 本地 Round 5 决胜推演。点击战队即可模拟胜者。</p>
        </div>
        <div class="matchup-controls" aria-label="Swiss controls">
          <span class="pick-count mono">${picks} PICKS</span>
          <button class="ghost-btn" type="button" data-swiss-undo ${picks ? "" : "disabled"}>↶ Undo</button>
          <button class="ghost-btn" type="button" data-swiss-reset ${picks ? "" : "disabled"}>⟲ Reset</button>
        </div>
      </div>
      <section class="swiss-round-board" role="region" tabindex="0" aria-label="Swiss 战绩池看板，可横向滚动">
        ${boardRounds.map((round, index) => `
          ${renderRoundColumn(round, simulation.selected_by_key, fixtureIndexByKey, index)}
          ${index < boardRounds.length - 1 ? renderRoundFlowArrow(round.round, boardRounds[index + 1].round) : ""}
        `).join("")}
      </section>
      ${renderPickemImpact(runtime)}
      <section class="standings-board" aria-label="战绩榜">
        <h2 class="sr-only">战绩榜 Standings</h2>
        ${renderRecordGroup("Advanced · 晋级", groups.advanced, "advanced")}
        ${renderRecordGroup("In Play · 存活", groups.live, "live")}
        ${renderRecordGroup("Eliminated · 淘汰", groups.eliminated, "eliminated")}
      </section>
    </div>
  `;
}

function renderFutureStage(stage, viewMode = "simple") {
  return `
    <div class="matchup-shell">
      ${renderStageControls(stage.stage_id, viewMode)}
      <section class="future-stage-page" aria-label="${escapeHtml(stage.stage_id)} status">
        <div class="future-stage-card hud">
          <span class="hud-c1"></span><span class="hud-c2"></span>
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

export function renderStageControls(currentStageId, viewMode = "simple") {
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
        <button class="${mode === "simple" ? "active" : ""}" type="button" title="Simple View" aria-label="Simple View" data-view-mode="simple">♛</button>
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

function renderRoundColumn(round, selectedByKey, fixtureIndexByKey, columnIndex = 0) {
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
    <section class="round-column reveal" style="--i:${columnIndex + 1}">
      <div class="round-heading">
        <h3>Round ${escapeHtml(round.round)}</h3>
        <span class="pool-label">${roundLabel(round)}</span>
        <span class="pool-count mono">${matches.length}</span>
      </div>
      <div class="round-stack">
        ${matches.length ? matches.join("") : `<div class="empty-round">WAITING<br>FOR DATA</div>`}
      </div>
    </section>
  `;
}

function renderRoundFlowArrow(fromRound, toRound) {
  return `
    <div class="round-flow-arrow" aria-hidden="true" title="Round ${escapeHtml(fromRound)} → Round ${escapeHtml(toRound)}">
      <span>▸</span>
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
        <span>${escapeHtml(options.locked ? "LOCKED" : "PICK")}</span>
        <span class="mono">${escapeHtml(matchMeta(match, options))}</span>
      </div>
    </div>
  `;
}

function renderSwissTeamButton(team, selected, options, match, side) {
  const accent = teamAccent(team);
  const logoFile = teamLogoFile(team);
  const logo = logoFile
    ? `<img class="team-logo" src="./assets/teams/${escapeHtml(logoFile)}.png" alt="" loading="lazy" decoding="async" onerror="this.remove()">`
    : "";
  const body = `<span class="team-mark team-${escapeHtml(teamSlug(team))}" aria-hidden="true">${logo}<span class="team-fallback">${escapeHtml(teamMark(team))}</span></span><span class="team-name">${escapeHtml(team)}</span>`;
  const style = ` style="--team:${accent}"`;
  if (options.locked || options.index === undefined) {
    const result = selected ? "获胜" : "落败";
    return `<div class="team-slot ${side}-slot ${selected ? "winner-slot" : "loser-slot"}"${style} aria-label="${escapeHtml(team)} ${result}">${body}</div>`;
  }
  return `<button class="team-slot ${side}-slot pick-slot ${selected ? "winner-slot" : ""}"${style} aria-label="模拟 ${escapeHtml(team)} 获胜" aria-pressed="${selected ? "true" : "false"}" data-index="${options.index}" data-winner="${escapeHtml(team)}">${body}</button>`;
}

/* ----- Pick'em objectives ----- */
export function renderPickemImpact(runtime) {
  if (!runtime || !runtime.rows || !runtime.rows.length) {
    return `
      <section class="pickem-dock">
        <div class="section-label">
          <div><span class="kicker">// PICK'EM OBJECTIVES</span><h2>Pick'em 兑现追踪</h2></div>
        </div>
        <div class="pickem-empty">等待 Pick 数据接入…</div>
      </section>
    `;
  }
  const s = runtime.summary;
  return `
    <section class="pickem-dock reveal" style="--i:6">
      <div class="section-label">
        <div><span class="kicker">// PICK'EM OBJECTIVES</span><h2>赛前提交答案单 · 兑现追踪</h2></div>
        <div class="objectives__summary">
          <span class="tally tally--locked mono">${s.locked} SECURED</span>
          <span class="tally tally--alive mono">${s.alive} LIVE</span>
          <span class="tally tally--broken mono">${s.broken} BROKEN</span>
        </div>
      </div>
      <div class="objectives__grid">
        ${runtime.rows.map(renderObjective).join("")}
      </div>
    </section>
  `;
}

function renderObjective(row) {
  const accent = teamAccent(row.team);
  const conf = clampPct(Number(row.confidence) * 100);
  const model = row.model || {};
  const votes = row.expert_votes || {};
  const cat = row.category;
  const next = nextStep(row);
  const stLabel = { locked: "SECURED", alive: "LIVE", broken: "BROKEN", missing: "N/A" }[row.status] || "—";

  const distRows = ["advance", "3-0", "0-3"].map((k) => {
    const pct = clampPct(Number(model[k] || 0) * 100);
    const label = k === "advance" ? "晋级" : k;
    return `
      <div class="dist__row" data-k="${k}">
        <span class="dist__key">${label}</span>
        <span class="bar"><span style="width:${pct}%"></span></span>
        <span class="dist__val mono">${pct}%</span>
      </div>`;
  }).join("");

  const totalVotes = (votes["advance"] || 0) + (votes["3-0"] || 0) + (votes["0-3"] || 0);
  const catVotes = votes[cat] || 0;
  const votePct = totalVotes ? Math.round((catVotes / totalVotes) * 100) : 0;
  const filled = Math.round(votePct / 10);
  const pips = Array.from({ length: 10 }, (_, i) => `<i class="${i < filled ? "on" : ""}"></i>`).join("");

  const logoFile = teamLogoFile(row.team);
  const logo = logoFile
    ? `<img src="./assets/teams/${escapeHtml(logoFile)}.png" alt="" loading="lazy" decoding="async" onerror="this.remove()">`
    : `<span class="team-fallback">${escapeHtml(teamMark(row.team))}</span>`;

  const fused = (Number(row.status_adjusted_score) || 0).toFixed(2);
  const mkt = clampPct(Number(row.market_win_prob_r1) * 100);
  const adv = clampPct(Number(model.advance) * 100);
  const sig = row.signals_agree != null ? `${row.signals_agree}/3` : "—";
  const tip = `融合分 <b class='mono'>${escapeHtml(fused)}</b> · 市场R1 <b class='mono'>${mkt}%</b> · 模型晋级 <b class='mono'>${adv}%</b> · 信号一致 <b class='mono'>${escapeHtml(sig)}</b>`;
  const srSummary = `${row.team}，类别 ${catLabel(cat)}，状态 ${stLabel}，融合置信度 ${conf}%，融合分 ${fused}，市场R1 ${mkt}%，模型晋级 ${adv}%，专家共识 ${votePct}%（${catVotes}/${totalVotes}），信号一致 ${sig}`;

  return `
    <article class="objective" role="group" data-st="${escapeHtml(row.status)}" style="--team:${accent}" data-tip="${tip}" aria-label="${escapeHtml(srSummary)}" tabindex="0">
      <div class="objective__head">
        <span class="objective__logo">${logo}</span>
        <span class="objective__id">
          <span class="objective__team">${escapeHtml(row.team)}</span>
          <span class="objective__tags">
            <span class="cat-badge">${escapeHtml(catLabel(cat))}</span>
            <span class="tier-badge">TIER ${escapeHtml(row.tier || "—")}</span>
          </span>
        </span>
        <span class="objective__st">${stLabel}</span>
      </div>
      <div class="objective__body">
        <div class="gauge" role="img" aria-label="融合置信度 ${conf}%">
          <svg viewBox="0 0 36 36">
            <circle class="gauge__track" cx="18" cy="18" r="15.915"></circle>
            <circle class="gauge__fill" cx="18" cy="18" r="15.915" stroke-dasharray="${conf} 100"></circle>
          </svg>
          <div class="gauge__val">${conf}<span class="gauge__cap">%</span></div>
        </div>
        <div class="dist"><span class="dist__cap">模型 · 边际概率</span>${distRows}</div>
      </div>
      <div class="objective__foot">
        <span class="votes-wrap">
          <span class="votes" aria-hidden="true" title="专家共识 ${votePct}% · ${catVotes}/${totalVotes}">${pips}</span>
          <span class="mono">${votePct}% 共识</span>
        </span>
        <span class="objective__next">${escapeHtml(next)}</span>
      </div>
    </article>
  `;
}

function catLabel(cat) {
  if (cat === "advance") return "ADVANCE";
  if (cat === "3-0") return "3-0";
  if (cat === "0-3") return "0-3";
  return String(cat || "—").toUpperCase();
}

function nextStep(row) {
  if (row.status === "locked") return "已兑现 ✓";
  if (row.status === "broken") return "已失效 ✕";
  if (row.status === "missing") return "无数据";
  const bits = [];
  if (row.wins_to_lock != null) bits.push(`+${row.wins_to_lock} 胜锁定`);
  if (row.losses_to_break != null) bits.push(`-${row.losses_to_break} 负失效`);
  if (row.losses_to_lock != null) bits.push(`-${row.losses_to_lock} 负锁定`);
  if (row.wins_to_break != null) bits.push(`+${row.wins_to_break} 胜失效`);
  return bits.join(" · ") || "进行中";
}

/* ----- Standings ----- */
export function renderRecordGroup(label, rows, grp) {
  return `
    <section class="record-group reveal" data-grp="${grp}" style="--i:7">
      <div class="section-label"><h3>${escapeHtml(label)}</h3><span class="group-count mono">${rows.length}</span></div>
      ${rows.length ? rows.map(renderStandingRow).join("") : `<div class="record-empty">— NONE —</div>`}
    </section>
  `;
}

function renderStandingRow(row) {
  const logoFile = teamLogoFile(row.team);
  const logo = logoFile
    ? `<img src="./assets/teams/${escapeHtml(logoFile)}.png" alt="" loading="lazy" decoding="async" onerror="this.remove()">`
    : `<span class="team-fallback">${escapeHtml(teamMark(row.team))}</span>`;
  const wins = Number(row.wins) || 0;
  const losses = Number(row.losses) || 0;
  const pips = Array.from({ length: 3 }, (_, i) => `<i class="${i < wins ? "w" : ""}"></i>`).join("") +
    Array.from({ length: 3 }, (_, i) => `<i class="${i < losses ? "l" : ""}"></i>`).join("");
  return `
    <div class="team-row" data-status="${escapeHtml(row.status)}">
      <span class="team-row__logo">${logo}</span>
      <span class="team-row__id">
        <span class="team-row__name">${escapeHtml(row.team)}</span>
        <span class="pips" aria-hidden="true">${pips}</span>
      </span>
      <span class="status-pill mono">${wins}-${losses}</span>
    </div>
  `;
}

/* ----- Bracket ----- */
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

/* ----- shared helpers ----- */
function groupFromRows(rows) {
  return {
    advanced: rows.filter((row) => row.status === "advanced"),
    live: rows.filter((row) => row.status === "alive"),
    eliminated: rows.filter((row) => row.status === "eliminated")
  };
}

function roundsForBoard(stage) {
  const byRound = new Map();
  const nestedResultRounds = new Set();
  for (const round of stage.rounds || []) {
    const key = String(round.round);
    byRound.set(key, {
      round: key,
      results: round.results || [],
      fixtures: round.fixtures || []
    });
    if ((round.results || []).length) nestedResultRounds.add(key);
  }
  for (const result of stage.results || []) {
    const key = String(result.round || "unknown");
    // nested round.results are authoritative; never let top-level results duplicate them
    if (nestedResultRounds.has(key)) continue;
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
  const labels = { "1": "0-0", "2": "1-0 / 0-1", "3": "2-0 / 1-1 / 0-2", "4": "2-1 / 1-2", "5": "2-2 DECIDER" };
  return labels[String(round.round)] || "SWISS";
}

function displayScore(match, options, selectedWinner) {
  if (!options.locked) {
    if (!selectedWinner) return "VS";
    return Number(match.best_of) === 1 ? "1 : 0" : "2 : 1";
  }
  const mapScores = String(match.map_scores || "");
  const score = mapScores && !mapScores.includes(";") ? mapScores : String(match.match_score || "");
  return score.replace(/\s*-\s*/g, " : ");
}

function matchMeta(match, options) {
  if (options.locked) {
    return String(match.maps || match.note || "").slice(0, 22);
  }
  return [match.team1_record, match.team2_record].filter(Boolean).join(" / ");
}

function matchStatus(match, options, selectedWinner) {
  const text = String(match.note || match.swiss_match_type || "").toLowerCase();
  if (!options.locked && String(match.swiss_round || match.round) === "5") {
    return { className: selectedWinner ? "advance-card" : "", label: selectedWinner ? "ADVANCE" : "", record: selectedWinner ? "→ S2" : "" };
  }
  if (text.includes("elimination")) return { className: "elimination-card", label: "ELIMINATED", record: "1:3" };
  if (text.includes("advancement")) return { className: "advance-card", label: "ADVANCE", record: "3:1" };
  return { className: "", label: "", record: "" };
}

function teamInitials(team) {
  return String(team || "?").split(/\s+/).filter(Boolean).map((part) => part[0]).join("").slice(0, 3).toUpperCase();
}

export function teamMark(team) {
  const marks = {
    "BetBoom": "BB", "FlyQuest": "FQ", "Gaimin Gladiators": "GG", "GamerLegion": "GL",
    "HEROIC": "H", "Liquid": "TL", "Lynn Vision": "LV", "Sharks": "SH",
    "SINNERS": "SIN", "THUNDER dOWNUNDER": "TD", "TYLOO": "TY"
  };
  return marks[team] || teamInitials(team);
}

export function teamLogoFile(team) {
  const logos = {
    "B8": "b8", "BIG": "big", "BetBoom": "betb", "FlyQuest": "fly", "Gaimin Gladiators": "gg",
    "GamerLegion": "gl", "HEROIC": "hero", "Liquid": "liqu", "Lynn Vision": "lvg", "M80": "m80",
    "MIBR": "mibr", "NRG": "nrg", "SINNERS": "sinn", "Sharks": "shks", "THUNDER dOWNUNDER": "tdu", "TYLOO": "tylo"
  };
  return logos[team] || "";
}

export function teamAccent(team) {
  const accents = {
    "B8": "#2f7bff", "BetBoom": "#ff4554", "BIG": "#cdd6e3", "FlyQuest": "#00d570",
    "Gaimin Gladiators": "#d4a02a", "GamerLegion": "#e8243b", "HEROIC": "#e01b38", "Liquid": "#2f74e0",
    "Lynn Vision": "#e0951a", "M80": "#d6f700", "MIBR": "#cdd6e3", "NRG": "#ff3901",
    "Sharks": "#2b5cff", "SINNERS": "#aab2c0", "THUNDER dOWNUNDER": "#9a3cff", "TYLOO": "#d82e26"
  };
  return accents[team] || "#ffb267";
}

export function teamSlug(team) {
  return String(team).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "team";
}

function resultKey(result) {
  const score = String(result.match_score || "").replace(/\s+/g, "").replace(/[–—]/g, "-");
  return [result.round || "round", result.team1 || "", result.team2 || "", result.winner || "", score].join(":");
}

export function clampPct(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function formatStamp(iso) {
  const s = String(iso || "");
  const m = s.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/);
  return m ? `${m[1]} ${m[2]} UTC` : (s || "—");
}

function sourceStatusClass(status) {
  if (status === "cached" || status === "fallback_success") return "status-warn";
  if (status === "failed") return "status-bad";
  return "status-good";
}

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}

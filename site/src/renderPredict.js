import {
  renderStageControls, renderModeSwitch, renderPickemImpact, renderRecordGroup,
  escapeHtml, teamLogoFile, teamMark, teamAccent
} from "./render.js";

/* ============================================================================
   PREDICT board — majors.im-style fully interactive Swiss bracket.
   Renders the live `buildSwiss` bracket as record-pool columns flowing into
   ADVANCED / ELIMINATED terminal columns; every match is pickable and the
   bracket re-pairs downstream on each pick.
   ========================================================================== */

export function renderPredictWorkspace(root, bracket, handlers, pickemRuntime = null, viewMode = "simple", meta = {}) {
  root.innerHTML = renderPredictHtml(bracket, pickemRuntime, viewMode, meta);

  root.querySelectorAll("[data-predict-mk]").forEach((btn) => {
    btn.addEventListener("click", () => handlers.onPredictPick?.(btn.dataset.predictMk, btn.dataset.predictTeam));
  });
  root.querySelectorAll("[data-view-mode]").forEach((btn) => {
    btn.addEventListener("click", () => handlers.onSwissViewMode?.(btn.dataset.viewMode));
  });
  root.querySelectorAll("[data-swiss-mode]").forEach((btn) => {
    btn.addEventListener("click", () => handlers.onSwissMode?.(btn.dataset.swissMode));
  });
  root.querySelector("[data-predict-undo]")?.addEventListener("click", () => handlers.onPredictUndo?.());
  root.querySelector("[data-predict-reset]")?.addEventListener("click", () => handlers.onPredictReset?.());
  root.querySelector("[data-predict-loadreal]")?.addEventListener("click", () => handlers.onPredictLoadReal?.());
}

export function renderPredictHtml(bracket, pickemRuntime, viewMode, meta) {
  const rounds = bracket.rounds || [];
  const standings = bracket.standings || [];
  const advanced = standings.filter((s) => s.status === "advanced");
  const eliminated = standings.filter((s) => s.status === "eliminated");
  const alive = standings.filter((s) => s.status === "alive");
  const pickCount = meta.pickCount ?? 0;
  const aligned = bracket.aligned !== false;

  const columns = rounds.map((rd, i) => `
    ${renderPredictColumn(rd, i)}
    ${renderFlowArrow()}
  `).join("");

  return `
    <div class="matchup-shell predict-shell view-${escapeHtml(viewMode)}">
      ${renderStageControls("stage-1", viewMode)}
      ${renderModeSwitch("predict")}
      <div class="matchup-header has-local-picks">
        <div class="matchup-header__title">
          <span class="kicker">// SWISS PREDICTOR · 自由推演</span>
          <h2>Predict the Swiss Stage</h2>
          <p>逐场点选你预测的胜者，战绩池自动按种子 / Buchholz 重排。${aligned
            ? `<b class="tag-real">与真实赛果一致</b>`
            : `<b class="tag-sim">已偏离真实赛果 · 引擎模拟配对</b>`}</p>
          <span class="predict-legend"><span class="lg lg-real">● REAL 真实赛果</span><span class="lg lg-sim">◌ SIM 引擎模拟配对</span></span>
        </div>
        <div class="matchup-controls" aria-label="Predict controls">
          <span class="pick-count mono">${advanced.length}/8 晋级 · ${pickCount} 选择</span>
          <button class="ghost-btn" type="button" data-predict-undo ${pickCount ? "" : "disabled"}>↶ Undo</button>
          <button class="ghost-btn" type="button" data-predict-loadreal>⭳ 真实赛果</button>
          <button class="ghost-btn" type="button" data-predict-reset ${pickCount ? "" : "disabled"}>✕ 清空</button>
        </div>
      </div>
      <section class="swiss-round-board predict-board" role="region" tabindex="0" aria-label="瑞士轮预测看板，可横向滚动">
        ${columns}
        ${renderTerminalColumn("Advanced", "advanced", "3-x", advanced, rounds.length + 1)}
        ${renderFlowArrow()}
        ${renderTerminalColumn("Eliminated", "eliminated", "x-3", eliminated, rounds.length + 2)}
      </section>
      ${renderPickemImpact(pickemRuntime)}
      <section class="standings-board" aria-label="战绩榜">
        <h2 class="sr-only">战绩榜 Standings</h2>
        ${renderRecordGroup("Advanced · 晋级", advanced, "advanced")}
        ${renderRecordGroup("In Play · 存活", alive, "live")}
        ${renderRecordGroup("Eliminated · 淘汰", eliminated, "eliminated")}
      </section>
    </div>
  `;
}

function renderPredictColumn(round, index) {
  const records = [...new Set(round.matches.map((m) => m.record))];
  const poolLabel = records.join(" / ") || "SWISS";
  const decided = round.matches.filter((m) => m.winner).length;
  return `
    <section class="round-column reveal" style="--i:${index + 1}">
      <div class="round-heading">
        <h3>Round ${escapeHtml(round.round)}</h3>
        <span class="pool-label">${escapeHtml(poolLabel)}</span>
        <span class="pool-count mono">${decided}/${round.matches.length}</span>
      </div>
      <div class="round-stack">
        ${round.matches.map(renderPredictCard).join("")}
      </div>
    </section>
  `;
}

function renderPredictCard(m) {
  const decided = !!m.winner;
  // only paint the mint 'advanced' accent once a winner is actually chosen
  const statusClass = decided && m.decider && (m.record.startsWith("2-") || m.record2.startsWith("2-")) ? "advance-card" : "";
  const isReal = !!(m.fromReal && m.realWinner);
  const seedOf = (t) => SEED[t] != null ? `#${SEED[t]}` : "";
  return `
    <div class="swiss-match-card predict-card ${decided ? "has-winner" : ""} ${statusClass} ${m.decider ? "is-decider" : ""}">
      <div class="match-ribbon">
        <span>BO${m.bo}${m.decider ? " · 决胜" : ""}</span>
        <strong class="ribbon-src ${isReal ? "src-real" : "src-sim"}" title="${isReal ? "与真实赛果一致" : "引擎模拟配对"}">${isReal ? "● REAL" : "◌ SIM"}</strong>
      </div>
      <div class="match-card-body">
        ${renderPredictTeam(m.team1, "left", m.winner === m.team1, decided, m.key)}
        <div class="match-score">${escapeHtml(predictScore(m, decided))}</div>
        ${renderPredictTeam(m.team2, "right", m.winner === m.team2, decided, m.key)}
      </div>
      <div class="match-card-meta">
        <span class="mono">${escapeHtml(m.record)}</span>
        <span class="mono">${escapeHtml(seedOf(m.team1))} · ${escapeHtml(seedOf(m.team2))}</span>
      </div>
    </div>
  `;
}

function renderPredictTeam(team, side, selected, decided, mk) {
  const accent = teamAccent(team);
  const logoFile = teamLogoFile(team);
  const logo = logoFile
    ? `<img class="team-logo" src="./assets/teams/${escapeHtml(logoFile)}.png" alt="" loading="lazy" decoding="async" onerror="this.remove()">`
    : "";
  const body = `<span class="team-mark" aria-hidden="true">${logo}<span class="team-fallback">${escapeHtml(teamMark(team))}</span></span><span class="team-name">${escapeHtml(team)}</span>`;
  const cls = selected ? "winner-slot" : (decided ? "loser-slot" : "");
  const label = !decided ? `预测 ${team} 获胜` : (selected ? `${team} · 已选为胜者` : `改选 ${team} 获胜`);
  return `<button class="team-slot ${side}-slot pick-slot ${cls}" style="--team:${accent}" aria-label="${escapeHtml(label)}" aria-pressed="${selected ? "true" : "false"}" data-predict-mk="${escapeHtml(mk)}" data-predict-team="${escapeHtml(team)}">${body}</button>`;
}

function renderTerminalColumn(label, status, poolLabel, rows, revealIndex = 6) {
  const sorted = rows.slice().sort((a, b) => (b.wins - a.wins) || (a.losses - b.losses) || (a.seed - b.seed));
  return `
    <section class="round-column term-column term-${status} reveal" style="--i:${revealIndex}" data-term="${status}">
      <div class="round-heading">
        <h3>${escapeHtml(label)}</h3>
        <span class="pool-label">${escapeHtml(poolLabel)}</span>
        <span class="pool-count mono">${rows.length}/8</span>
      </div>
      <div class="round-stack">
        ${sorted.length
          ? sorted.map((r) => renderTermChip(r, status)).join("")
          : `<div class="empty-round">${status === "advanced" ? "尚无晋级" : "尚无淘汰"}</div>`}
      </div>
    </section>
  `;
}

function renderTermChip(row, status) {
  const accent = teamAccent(row.team);
  const logoFile = teamLogoFile(row.team);
  const logo = logoFile
    ? `<img src="./assets/teams/${escapeHtml(logoFile)}.png" alt="" loading="lazy" decoding="async" onerror="this.remove()">`
    : `<span class="team-fallback">${escapeHtml(teamMark(row.team))}</span>`;
  return `
    <div class="term-chip" data-status="${status}" style="--team:${accent}">
      <span class="term-chip__logo">${logo}</span>
      <span class="term-chip__name">${escapeHtml(row.team)}</span>
      <span class="term-chip__rec status-pill mono">${row.wins}-${row.losses}</span>
    </div>
  `;
}

function renderFlowArrow() {
  return `<div class="round-flow-arrow" aria-hidden="true"><span>▸</span></div>`;
}

function predictScore(m, decided) {
  if (!decided) return "VS";
  // real, unchanged match -> show the true series score oriented to the winner
  if (m.realScore && m.realWinner === m.winner) {
    const parts = String(m.realScore).split(/[-:]/).map((s) => s.trim());
    if (parts.length === 2) return m.winner === m.team1 ? `${parts[0]} : ${parts[1]}` : `${parts[1]} : ${parts[0]}`;
  }
  return "✓"; // simulated pick: decided, but no fabricated scoreline
}

const SEED = {
  "GamerLegion": 1, "B8": 2, "BetBoom": 3, "MIBR": 4, "HEROIC": 5, "Lynn Vision": 6,
  "BIG": 7, "TYLOO": 8, "SINNERS": 9, "M80": 10, "Liquid": 11, "Sharks": 12,
  "NRG": 13, "Gaimin Gladiators": 14, "THUNDER dOWNUNDER": 15, "FlyQuest": 16
};

# 瑞士轮 PREDICT 选择器 majors.im 操作手感 · 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`).

**Goal:** 给 `site/` 的 PREDICT 瑞士轮选择器加上 majors.im 式操作手感——悬停追踪、FLIP 重排动画、可操作引导、冲突提示、终列联动——不改引擎、不碰 Python 栈。

**Architecture:** 纯逻辑放新模块 `site/src/trace.js`(顶层不碰 `document`，可单测)；DOM 副作用全部进 `effects.js`；`renderPredict.js` 只加渲染钩子(稳定 key、终列 `data-predict-team`、`.is-actionable`)；`main.js` 在 `renderPredictBoard` 前后串 FLIP capture/play 与冲突 diff；视觉与降级在 `styles.css`。

**Tech Stack:** Vanilla ES Modules(无构建)、`node --test`、GSAP(CDN，特性检测)、`prefers-reduced-motion` 门控。

---

### Task 1: 纯逻辑模块 `trace.js`(traceTeam / computeFlipTransforms / diffPrunedPicks)

**Files:**
- Create: `site/src/trace.js`
- Test: `site/tests/trace.test.mjs`

- [ ] **Step 1: 写失败测试** `site/tests/trace.test.mjs`

覆盖：traceTeam 收集某队所有对阵 matchKey + 终列状态；空/未知队鲁棒；computeFlipTransforms 只对移动的 key 返回 dx/dy、跳过无匹配 key；diffPrunedPicks 找出 matchup 已不存在或 winner 不再是参与者的 pick、保留有效 pick。测试用真实 `buildSwiss({})` 输出做夹具。

- [ ] **Step 2: 跑测试看失败** — `node --test site/tests/trace.test.mjs` → FAIL(模块不存在)

- [ ] **Step 3: 实现 `site/src/trace.js`**

```js
// Pure logic for the PREDICT selector UX layer. No document at module top-level.
export function traceTeam(bracket, team) {
  const matchKeys = new Set();
  if (!bracket || !team) return { matchKeys, terminal: "alive" };
  for (const rd of bracket.rounds || []) {
    for (const m of rd.matches || []) {
      if (m.team1 === team || m.team2 === team) matchKeys.add(m.key);
    }
  }
  let terminal = "alive";
  for (const s of bracket.standings || []) {
    if (s.team === team) { terminal = s.status; break; }
  }
  return { matchKeys, terminal };
}

export function computeFlipTransforms(oldRects, newRects) {
  const out = [];
  for (const key of Object.keys(newRects || {})) {
    const a = (oldRects || {})[key];
    const b = newRects[key];
    if (!a || !b) continue;                 // unmatched -> fade, not flip
    const dx = a.left - b.left;
    const dy = a.top - b.top;
    if (dx === 0 && dy === 0) continue;
    out.push({ key, dx, dy });
  }
  return out;
}

export function diffPrunedPicks(prevPicks, bracket) {
  const teamsByKey = new Map();
  for (const rd of (bracket && bracket.rounds) || []) {
    for (const m of rd.matches || []) teamsByKey.set(m.key, [m.team1, m.team2]);
  }
  const pruned = [];
  for (const [k, v] of Object.entries(prevPicks || {})) {
    const teams = teamsByKey.get(k);
    if (!teams || !teams.includes(v)) {
      pruned.push({ key: k, team: v, label: String(k).split(" :: ").join(" vs ") });
    }
  }
  return pruned;
}
```

- [ ] **Step 4: 跑测试看通过** — `node --test site/tests/trace.test.mjs` → PASS
- [ ] **Step 5: 跑全套回归** — `node --test site/tests/*.test.mjs` → 30 + 新增 全绿
- [ ] **Step 6: commit** — `feat(site): pure trace/flip/conflict logic for predict selector`

---

### Task 2: 渲染钩子(`renderPredict.js`)——FLIP 稳定 key、终列 `data-predict-team`、`.is-actionable`

**Files:**
- Modify: `site/src/renderPredict.js`(renderPredictCard:99、renderTermChip:154)
- Test: `site/tests/predict.test.mjs`(扩断言)

- [ ] **Step 1: 写失败测试**(predict.test.mjs)：断言每张 `predict-card` 带 `data-mk`(= m.key)供 FLIP 配对；尚未定胜负且双方已知的卡带 `is-actionable`；终列 chip 带 `data-predict-team`。
- [ ] **Step 2: 跑看失败**
- [ ] **Step 3: 实现** — renderPredictCard 容器加 `data-mk="${escapeHtml(m.key)}"` 与 `${!decided && m.team1 && m.team2 ? "is-actionable" : ""}` 类；renderTermChip 容器加 `data-predict-team="${escapeHtml(row.team)}"`。**不删任何现有类/属性**。
- [ ] **Step 4: 跑看通过** + 全套回归全绿
- [ ] **Step 5: commit** — `feat(site): predict-card flip key + actionable + terminal trace hooks`

---

### Task 3: 悬停追踪(`effects.js`)——hover-to-trace

**Files:**
- Modify: `site/src/effects.js`(新增 `initPredictTrace(root, bracket)`，由 main 调用)
- Modify: `site/src/main.js`(renderPredictBoard 末尾调用)

- [ ] **Step 1: 实现 `initPredictTrace(root, bracket)`** — 事件委托 `mouseover`/`mouseout`/`focusin`/`focusout` 于 `predict-board`：从 `e.target.closest("[data-predict-team]")` 取队名 → `traceTeam(bracket, team)` → 给 `data-mk ∈ matchKeys` 的卡加 `.is-traced`、board 加 `.has-trace`；leave 清除。特性检测：无 board 直接返回。纯 import `traceTeam`。
- [ ] **Step 2: main.js 接线** — `renderPredictBoard` 末尾(afterRender 后)`initPredictTrace(root, bracket)`。
- [ ] **Step 3: 全套回归全绿**(DOM 行为靠 trace.js 单测 + 手动验证) + commit `feat(site): hover-to-trace highlight on predict board`

---

### Task 4: FLIP 重排动画(`effects.js` + `main.js`)

- [ ] **Step 1: 实现 `effects.js` 的 `captureRects(root)`/`playFlip(root, oldRects)`** — captureRects 读 `[data-mk],[data-predict-team]` 的 `getBoundingClientRect()` 存 `{left,top}`；playFlip 重渲后再测 → `computeFlipTransforms` → GSAP `fromTo` 把元素从旧偏移滑回 0(`window.gsap` 特性检测，`REDUCE` 或无 gsap 时跳过)。
- [ ] **Step 2: main.js 接线** — `onPredictPick` 在重渲前 `const old = captureRects(root)`，`renderPredictBoard` 后 `playFlip(root, old)`(仅交互触发，初次加载不播)。
- [ ] **Step 3: 全套回归全绿** + 本地预览人工确认动画 + reduced-motion 降级 + commit `feat(site): FLIP reorder animation on pick`

---

### Task 5: 冲突提示(`effects.js` + `main.js`)

- [ ] **Step 1: 实现 `effects.js` 的 `showConflictToast(dropped)`** — 复用 `#tip` 或新增 `.predict-toast` 容器，`role="status" aria-live="polite"`，文案如 `已失效：${labels.join("、")}`，1.8s 自动消失。
- [ ] **Step 2: main.js 接线** — `onPredictPick` 重渲前 `const before = {...predictPicks}`；裁剪后用 `diffPrunedPicks(before, bracket)` 得 dropped，非空则 `showConflictToast(dropped)`。
- [ ] **Step 3: 全套回归全绿** + commit `feat(site): non-silent conflict toast when picks are pruned`

---

### Task 6: 视觉与降级(`styles.css`)

- [ ] **Step 1: 加样式** — `.has-trace .predict-card:not(.is-traced){opacity:.32;filter:saturate(.4)}`、`.predict-card.is-traced{...高亮描边}`、`.predict-card.is-actionable{...微光/脉冲}`、`.predict-toast{...}`；全部 `@media (prefers-reduced-motion: reduce)` 降级为静态(无脉冲/无过渡)。
- [ ] **Step 2: 全套回归全绿** + 本地预览三态(hover/可操作/toast)人工确认 + commit `style(site): trace/actionable/toast visuals + reduced-motion`

---

### Task 7: 集成验证

- [ ] **Step 1: 全套测试** — `node --test site/tests/*.test.mjs` 全绿
- [ ] **Step 2: 本地预览** — `python3 -m http.server 8000 --directory site`，逐一验证五特性 + 现有 PREDICT/LIVE 切换/Undo/Reset/真实赛果不回归。
- [ ] **Step 3: 截图存档(可选)** + 最终 commit。

## 契约保护清单(每步后复跑确认)
保留：`swiss-round-board`/`predict-board`、`round-flow-arrow`、`pickem-dock`、`team-mark`/`team-logo`、`match-score`、`locked-match`/`fixture-match`、`data-winner`、`data-predict-mk`/`data-predict-team`、`term-advanced`/`term-eliminated`、`predict-card`、view-switcher 按钮属性顺序、stage 链接。`index.html` 含 `IEM Cologne Major 2026 Simulator`。
新增：`data-mk`、`.is-traced`/`.has-trace`/`.is-actionable`、终列 `data-predict-team`、`.predict-toast`。

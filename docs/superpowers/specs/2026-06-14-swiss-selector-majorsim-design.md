# 瑞士轮 PREDICT 选择器 — majors.im 操作手感复刻 · 设计

- 日期：2026-06-14
- 范围：前端 `site/`（零构建纯 ES Module 站点），不触碰 `src/`（Python ML 栈）
- 状态：已与用户对齐，待写实现计划

## 1. 背景与目标

`site/` 的 PREDICT 模式已具备 majors.im 的**内核**：逐场点选胜者 → 按种子 + Buchholz 即时把下游瑞士轮全网重排（`swissSim.buildSwiss` 纯函数，确定性）。用户反馈"不够易用，需要复刻 majors.im 的操作逻辑"——差距不在引擎，而在 **操作反馈与导航的丝滑层**。

本设计复刻的是 majors.im 的**功能性交互模式**（hover 追踪、平滑重排、操作反馈），用项目自有实现与现有暗色战术终端视觉，不照搬其代码、视觉资产或文案。

## 2. 现状测绘摘要

- **交互流**：点队伍 slot（`data-predict-mk` + `data-predict-team`）→ `main.js:onPredictPick(mk, team)` toggle 写入 `predictPicks` → `renderPredictBoard(false)` → `buildSwiss(predictPicks, realData)` 纯函数重算 → 整页重画字符串 → DOM 替换 + 事件重绑。
- **已有**：Undo（`predictOrder` 回退）、Reset、Load Real。REAL/SIM ribbon。
- **痛点**：① 每次点击整页瞬间重画、无过渡，看不清谁被重排去了哪；② 悬停队伍零反馈；③ 选择因重排失效时被**静默删除**；④ 只能逐场点、缺当前可操作引导。
- **架构铁律**（`CLAUDE.md`）：`render.js`/`renderPredict.js` 为纯字符串渲染 + DOM 安全，**模块顶层不得引用 `document`**（单测用 fakeRoot import）；DOM 副作用集中在 `effects.js`；逻辑纯函数在 `swissSim.js`/`swiss.js`/`pickem.js`/`bracket.js`。
- **引擎接口**：`buildSwiss(picks, opts) → { rounds, states, standings, picks, validKeys, complete, aligned }`，无副作用、幂等。

## 3. 设计：五个核心特性

### 3.1 悬停即追踪（hover-to-trace）— majors.im 最标志性的易读特征
- **行为**：鼠标悬停任一支队（任意轮的 slot 或终列里的队）→ 高亮该队在每一轮参与的对阵卡 + 其晋级/淘汰落点；其余卡片淡出（dim）。`mouseleave` 恢复。
- **纯逻辑**（可单测）：新增 `traceTeam(bracket, team) → { matchKeys: Set<string>, terminal: 'advanced'|'eliminated'|'alive' }`，从 `buildSwiss` 输出推导该队涉及的全部 matchKey 与终列状态。
- **DOM**（`effects.js`）：事件委托 `mouseover`/`mouseout` 读 hover 元素的 `data-predict-team`，对命中 `matchKeys` 的卡加 `.is-traced`、board 加 `.has-trace`（CSS 据此把非 traced 卡 dim）。纯 CSS 控制视觉，避免逐元素 JS 改样式。
- **性能**：单次委托 + 类切换，O(卡片数)。

### 3.2 重排有动画（FLIP），不再整页闪变
- **行为**：点选胜者后，队伍/卡片从旧位置**平滑滑到新位置**（进下一轮、对上新对手、落入终列），而非瞬间重画。
- **机制**：FLIP（First-Last-Invert-Play）。重渲染前按稳定 key（卡片 `data-predict-mk`、队伍 `data-predict-team`）测量旧位置（First）→ 重渲染后测新位置（Last）→ GSAP 从 `Δ` 反向位移到 0（Invert + Play）。
- **纯逻辑**（可单测）：`computeFlipTransforms(oldRects, newRects) → [{ key, dx, dy }]`，纯几何 diff，与 DOM 解耦。
- **DOM**（`effects.js`）：在 `onPredictPick → renderPredictBoard` 前后插 capture/play hook。GSAP 特性检测；`prefers-reduced-motion` 或 GSAP 缺失（CDN 离线）→ 跳过动画直接呈现最终态。
- **约束**：动画纯视觉，不改变 `buildSwiss` 最终 DOM 结构与契约。

### 3.3 当前可操作引导
- **行为**：尚未定胜负且双方已知的"可点"卡给轻微视觉强调（微光/脉冲）；已定卡收敛。让你一眼知道"接下来该选哪几场"。
- **实现**：`renderPredict.js` 给符合条件的卡加 `.is-actionable` 类（纯渲染判定，无 DOM 副作用）；`styles.css` 定义强调，`prefers-reduced-motion` 下降级为静态描边。

### 3.4 冲突不再静默
- **行为**：当一次点击导致此前某些 picks 因重排失效被裁剪，弹出一闪而过的提示（如"TYLOO vs FaZe 已不会发生"），而非默默消失。
- **纯逻辑**（可单测）：`diffPrunedPicks(prevPicks, validKeys) → string[]`（被裁掉的 matchKey 列表）。已有 `validKeys` 裁剪逻辑，只需把差集提取出来。
- **DOM**（`effects.js`）：复用页面已有的浮层 `#tip` 或新增轻量 toast 呈现，自动消失；可达性 `role="status"` `aria-live="polite"`。

### 3.5 终列与战绩实时联动
- **行为**：右侧"3-0 晋级 / 0-3 淘汰"终列（`term-advanced`/`term-eliminated`）里的队同样响应 hover-trace（3.1）。
- **实现**：终列队伍元素补挂 `data-predict-team`，纳入 3.1 的委托与 `traceTeam` 命中集。

## 4. 模块边界

- **新增** `site/src/trace.js` — 纯逻辑，顶层不引用 `document`：`traceTeam`、`computeFlipTransforms`、`diffPrunedPicks`。可 fakeRoot/直接 import 单测。
- **`effects.js`** — 承接所有 DOM 副作用：hover 高亮类切换、GSAP FLIP、toast。
- **`renderPredict.js`** — 仅加渲染钩子：终列队补 `data-predict-team`、可点卡加 `.is-actionable`。保持纯字符串渲染、顶层不碰 `document`。
- **`main.js`** — `onPredictPick` 串入 FLIP capture/play 与冲突 diff 调用。
- **`styles.css`** — `.is-traced`/`.has-trace`/`.is-actionable`/toast 的视觉与 `prefers-reduced-motion` 降级。

## 5. 渲染契约保护

**保留（测试已锁定，只新增、不删除）**：`swiss-round-board`、`predict-board`、`round-flow-arrow`、`pickem-dock`、`team-mark`、`team-logo`、`match-score`、`locked-match`、`fixture-match`、`data-winner`、`data-predict-mk`、`data-predict-team`、`term-advanced`、`term-eliminated`、`predict-card`、view-switcher 按钮属性顺序、stage 链接。`index.html` 保留 `IEM Cologne Major 2026 Simulator`，不引入 `AI Desk`/`Model Lab`/`Overview`。

**新增**：`.is-traced`、`.has-trace`、`.is-actionable` 类；终列队 `data-predict-team`；toast 容器。

## 6. 测试策略

- **新增** `site/tests/trace.test.mjs`：单测 `traceTeam`（含多轮路径、终列、未参与队）、`computeFlipTransforms`（位置 diff、缺失 key、reduced 情形）、`diffPrunedPicks`（裁剪差集、空集）。
- **现有 30 项保持全绿**：`node --test site/tests/*.test.mjs`；`render.test.mjs`/`predict.test.mjs`/`shell.test.mjs` 契约不破。
- **渲染钩子**：在现有 fakeRoot 风格下断言 `.is-actionable`、终列 `data-predict-team` 的存在性。
- **DOM 动效**：GSAP/hover 的副作用部分以纯逻辑覆盖为主 + 本地 http 预览人工验证（`python3 -m http.server 8000 --directory site`）。

## 7. 非目标（YAGNI）

- 键盘操作（方向键/Enter 选胜者）—— 明确不做。
- 终列 force 战绩快选（点空位 force 某队 3-0/0-3）—— 不做。
- LIVE 模式数据流改动；瑞士轮配对引擎重写（`buildSwiss` 即时重排已是内核，保持）。
- 任何 `src/`（Python ML）改动。

## 8. 文件影响清单

| 文件 | 改动 |
| --- | --- |
| `site/src/trace.js` | 新增纯逻辑模块（traceTeam/computeFlipTransforms/diffPrunedPicks） |
| `site/src/effects.js` | hover 高亮委托、GSAP FLIP、冲突 toast |
| `site/src/renderPredict.js` | 终列 `data-predict-team`、`.is-actionable` 钩子 |
| `site/src/main.js` | onPredictPick 串 FLIP capture/play + 冲突 diff |
| `site/styles.css` | trace/actionable/toast 视觉 + reduced-motion 降级 |
| `site/tests/trace.test.mjs` | 新增纯逻辑单测 |

## 9. 风险与缓解

- **FLIP key 匹配**：重渲前后用稳定 key（`data-predict-mk`/`data-predict-team`）配对；无匹配的元素淡入/淡出而非位移。
- **离线/降级**：GSAP 经 CDN，特性检测；缺失或 `prefers-reduced-motion` → 无动画直达最终态，功能不受损。
- **悬停性能**：事件委托 + CSS 类，避免逐元素 JS 样式写入。
- **契约回归**：先跑现有 30 项基线，每步改动后复跑保持全绿。

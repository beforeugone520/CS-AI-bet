# 项目指令

## 语言偏好

- 用中文进行思考（思维链 / reasoning）。
- 用中文回复用户。

## 项目结构（两块相对独立）

- **Python ML 工具链** `src/cs2pickem/` —— CS2 Major Pick'em 预测建模 / CLI。开发约定见根目录 `AGENTS.md`（加速依赖、纯 Python 回退、`CS2PICKEM_ACCELERATED_MLP`）。测试：`PYTHONPATH=src python3 -m unittest discover -s tests -v`（当前 512 项）。
  - **WF-2 建模升级现状**：生产默认已采纳 **Glicko-2 赛前评级**（`MatchPredictor.train` 默认 `rating_mode='glicko'`，Elo 并存）+ **logit_pool 市场融合**（`model_weight≈0.30`，`forecast`/`pickem` 两条生产路径显式传参）。其余建模轴（Bradley-Terry 先验 `inject_bt`、多方法校准 `calibration_method`、`include_unverified`、`devig_method` 变体、Glicko-MOV）**实现但保持默认关闭、仅 opt-in**——WF-2F 大回测判定它们 no_significant_diff / 回归 / 不可证伪，详见 `docs/modeling-upgrade-2026-06.md`。
  - **底层常量刻意不动**：`strategy.DEFAULT_FUSION_METHOD='legacy_clip'` / `DEFAULT_MODEL_WEIGHT=0.35` 与 `tuning._DEFAULT_RATING_MODE='elo'` 被行为契约测试与 A/B 同口径基线锁定，**故意不改**；生产翻转通过调用点显式传 `strategy.PRODUCTION_FUSION_METHOD` / `PRODUCTION_MODEL_WEIGHT` 实现。后续改动**勿擅自改这些全局常量或其他默认**——只翻有证据支持的默认是 WF-2F 红线。
- **前端静态站** `site/` —— 零构建纯 ES Module 站点（战术情报终端 UI + majors.im 式瑞士轮预测器），部署到 GitHub Pages。**与 Python 栈无耦合**：纯前端/视觉改动不应触碰 `src/`、`tests/`、`scripts/`。

## 前端约定（site/）

- 测试：`node --test site/tests/*.test.mjs`（提交前须全绿，当前 30 项）。本地预览：`python3 -m http.server 8000 --directory site`（需 http，`fetch` 不支持 `file://`）。
- `render.js` / `renderPredict.js` 为**纯字符串渲染 + DOM 安全**（单测用 fakeRoot 直接 import）：模块顶层不得引用 `document`。`render.test.mjs` / `shell.test.mjs` 锁定了一批契约——保留这些类名/属性：`swiss-round-board`、`round-flow-arrow`、`pickem-dock`、`team-mark`、`team-logo`、`match-score`、`locked-match`、`fixture-match`、`data-winner`、view-switcher 按钮属性顺序 `class`→`type`→`title`、stage 链接；`index.html` 须含 `IEM Cologne Major 2026 Simulator` 且不含 `AI Desk` / `Model Lab` / `Overview`。
- 逻辑纯函数：`swissSim.js`（瑞士轮引擎，混合真实/模拟，移植自 `src/cs2pickem/swiss.py`）、`swiss.js`、`pickem.js`、`bracket.js`。瑞士看板两种模式：**PREDICT**（默认，全交互预测器）/ **LIVE**（真实赛果）。
- 第三方库（GSAP、Google Fonts）经 CDN 引入并**特性检测、离线优雅降级**；所有动效都用 `prefers-reduced-motion` 门控。
- 部署：push `main` → `.github/workflows/pages.yml` → `gh-pages` → <https://beforeugone520.github.io/CS-AI-bet/>。仅 `site/**`、`scripts/*site*`、`data/cologne2026/**`、workflow 改动会触发重部署（README / CLAUDE.md 等不会）。

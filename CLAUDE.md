# 项目指令

## 语言偏好

- 用中文进行思考（思维链 / reasoning）。
- 用中文回复用户。

## 项目结构（两块相对独立）

- **Python ML 工具链** `src/cs2pickem/` —— CS2 Major Pick'em 预测建模 / CLI。开发约定见根目录 `AGENTS.md`（加速依赖、纯 Python 回退、`CS2PICKEM_ACCELERATED_MLP`）。测试：`PYTHONPATH=src python3 -m unittest discover -s tests -v`。
- **前端静态站** `site/` —— 零构建纯 ES Module 站点（战术情报终端 UI + majors.im 式瑞士轮预测器），部署到 GitHub Pages。**与 Python 栈无耦合**：纯前端/视觉改动不应触碰 `src/`、`tests/`、`scripts/`。

## 前端约定（site/）

- 测试：`node --test site/tests/*.test.mjs`（提交前须全绿，当前 30 项）。本地预览：`python3 -m http.server 8000 --directory site`（需 http，`fetch` 不支持 `file://`）。
- `render.js` / `renderPredict.js` 为**纯字符串渲染 + DOM 安全**（单测用 fakeRoot 直接 import）：模块顶层不得引用 `document`。`render.test.mjs` / `shell.test.mjs` 锁定了一批契约——保留这些类名/属性：`swiss-round-board`、`round-flow-arrow`、`pickem-dock`、`team-mark`、`team-logo`、`match-score`、`locked-match`、`fixture-match`、`data-winner`、view-switcher 按钮属性顺序 `class`→`type`→`title`、stage 链接；`index.html` 须含 `IEM Cologne Major 2026 Simulator` 且不含 `AI Desk` / `Model Lab` / `Overview`。
- 逻辑纯函数：`swissSim.js`（瑞士轮引擎，混合真实/模拟，移植自 `src/cs2pickem/swiss.py`）、`swiss.js`、`pickem.js`、`bracket.js`。瑞士看板两种模式：**PREDICT**（默认，全交互预测器）/ **LIVE**（真实赛果）。
- 第三方库（GSAP、Google Fonts）经 CDN 引入并**特性检测、离线优雅降级**；所有动效都用 `prefers-reduced-motion` 门控。
- 部署：push `main` → `.github/workflows/pages.yml` → `gh-pages` → <https://beforeugone520.github.io/CS-AI-bet/>。仅 `site/**`、`scripts/*site*`、`data/cologne2026/**`、workflow 改动会触发重部署（README / CLAUDE.md 等不会）。

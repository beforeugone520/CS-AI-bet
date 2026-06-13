<div align="center">

<img src="docs/images/site-hero.png" width="900" alt="MAJORS.IM — IEM Cologne Major 2026 Pick'em 战术情报终端" />

# CS2 Major Pick'em · 战术情报终端

**给玩家与分析师的 IEM Cologne Major 2026 瑞士轮 Pick'em 指挥中心**
赛果锁定 · 浏览器内本地推演 · 模型 / 专家 / 市场三路融合 · 兑现实时追踪

<br>

[![Counter-Strike 2](https://img.shields.io/badge/Counter--Strike_2-IEM_Cologne_Major_2026-ff5722?style=for-the-badge&labelColor=0a0908)](https://esports.gg/news/counter-strike-2/iem-cologne-major-2026-stage-1-overview-results/)
[![Static Site](https://img.shields.io/badge/Static_Site-GitHub_Pages-46d2ff?style=for-the-badge&labelColor=0a0908)](#-静态网站部署)
[![License: MIT](https://img.shields.io/badge/License-MIT-4ee6a8?style=for-the-badge&labelColor=0a0908)](LICENSE)

![Vanilla JS](https://img.shields.io/badge/Vanilla_JS-ESM_无构建-f7df1e?style=flat-square&labelColor=0a0908)
![GSAP](https://img.shields.io/badge/GSAP-3.15-88ce02?style=flat-square&labelColor=0a0908)
![Hand--rolled SVG](https://img.shields.io/badge/可视化-手写_SVG-ff8a2e?style=flat-square&labelColor=0a0908)
![Python ML](https://img.shields.io/badge/建模-scikit--learn_·_xgboost-3776ab?style=flat-square&labelColor=0a0908)
![Reduced Motion](https://img.shields.io/badge/A11y-prefers--reduced--motion-9a3cff?style=flat-square&labelColor=0a0908)

<br>

[**🎯 在线指挥中心**](#-静态网站部署) ·
[数据处理与建模](docs/data-processing.md) ·
[赛前答案 JSON](data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/final_fused_pickem_2026-06-01.json) ·
[Round 4 检查点](data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/final_fused_pickem_checkpoint_round4_2026-06-05.json)

</div>

---

## ✨ 这是什么

`site/` 是一个**零构建、纯 ES Module** 的静态站点，把 IEM Cologne Major 2026 的瑞士轮做成一块 **broadcast 级战术情报终端**：

- **🟢 赛果已锁定** —— Round 1–4 的真实 BO3 结果直接写死，状态色一眼可读（薄荷=晋级 / 电光青=存活 / 绯红=出局 / 金=2-2 决胜）。
- **🎮 本地推演** —— 点击任意未完成对局的战队即可在浏览器内模拟胜负，战绩榜与 Pick'em 兑现状态实时联动，支持 Undo / Reset。
- **📊 三路融合情报** —— 每个 Pick'em 槽位都展开为一张情报卡：融合置信度雷达、模型分布条（晋级 / 3-0 / 0-3）、专家投票分布、市场对比 edge。
- **🛰️ 四种视图** —— Simple / Minimal / Bracket / Classic，一键切换信息密度。

> 仓库快照停在 **2026-06-05 Round 5 开赛前**。Round 4 赛果已收齐，Stage 1 未完赛前只写 checkpoint，不提前回测最终 Pick'em。

<br>

<div align="center">

### 🗺️ 瑞士轮战绩池看板

<img src="docs/images/site-board.png" width="880" alt="瑞士轮战绩池看板：按 0-0 → 1-0/0-1 → 2-0/1-1/0-2 → 2-1/1-2 → 2-2 决胜 左右推进" />

### 🎯 Pick'em 情报卡 · 兑现追踪

<img src="docs/images/site-objectives.png" width="880" alt="Pick'em 情报卡：融合置信度雷达 + 模型分布条 + 专家投票分布 + 状态" />

</div>

---

## 📑 目录

| 想看什么 | 跳转 |
| --- | --- |
| 视觉与交互设计 | [设计语言](#-设计语言--战术情报终端) |
| 赛前提交的 Pick'em 答案单 | [赛前提交清单](#-赛前提交清单) |
| Round 4 后兑现进度 | [Round 4 后进度](#-round-4-后进度) |
| Round 5 关键赛程 | [Round 5 关键赛程](#-round-5-关键赛程) |
| 复盘读数与诊断 | [复盘读数](#-复盘读数) |
| 本地预览与部署 | [静态网站部署](#-静态网站部署) |
| 数据清洗 / 特征 / 模型 / 回测 | [技术二级菜单](#-技术二级菜单) |

---

## 🎨 设计语言 · 战术情报终端

整个界面是一间 **「余烬色作战室」**：近黑底色，单一琥珀/火焰强调色，配合严格的状态语义色，所有数字锁进等宽字体像 Bloomberg 终端一样对齐。

<table>
<tr>
<td width="50%" valign="top">

**🎨 视觉系统**
- 五层氛围背景：余烬光晕 + 战术网格 + feTurbulence 噪点 + 暗角 + 雷达粒子画布
- 玻璃面板（`backdrop-filter` + 顶部高光「点亮边」）
- L 形战术角框，悬停「目标锁定」外扩
- 逐队官方品牌色微调每张卡片

</td>
<td width="50%" valign="top">

**🧩 字体与数据**
- `Chakra Petch` 战术标题 · `Outfit` 正文 · `JetBrains Mono` 数据
- 全数字 `tabular-nums slashed-zero`，计数不抖动
- 手写内联 SVG 雷达仪表盘 / 概率条 / 投票 pip
- 状态色只承载语义，绝不当装饰滥用

</td>
</tr>
</table>

**⚡ 交互与动效**（全部 `prefers-reduced-motion` 门控，并对 CDN / 离线优雅降级）
GSAP 计数入场 · 分层 stagger 揭示 · 悬停目标锁定 · 选择即时联动战绩榜 · 情报卡悬浮 tooltip · 琥珀 / 极地双主题切换 · 雷达粒子氛围画布。

<div align="center">
<img src="docs/images/site-mobile.png" width="260" alt="移动端响应式布局" />

<sub>移动端：hero 堆叠、看板横向滚动、情报卡纵向铺开</sub>
</div>

---

## 📋 赛前提交清单

这是 2026-06-01 的赛前最终融合答案单。融合权重为专家 `0.30`、市场 `0.20`、模型 `0.50`，并把选手样本 / 替补风险写入 `status_adjusted_score`。

| 槽位 | 队伍 | Round 4 后状态 |
| --- | --- | --- |
| `3-0` | MIBR, GamerLegion | 两队都已晋级 Stage 2，但均为 `3-1`，两个 `3-0` 槽位已失效 |
| `晋级` | BIG, BetBoom, B8, HEROIC, M80, TYLOO | BetBoom、B8、M80 已兑现；BIG、TYLOO 进入 Round 5 仍可兑现；HEROIC 已失效 |
| `0-3` | NRG, Gaimin Gladiators | Gaimin Gladiators 已命中；NRG 已失效 |

> 上表是赛前提交快照，不会因为赛后结果反向修改。

---

## 🛰️ Round 4 后进度

2026-06-04 Round 4 六场 BO3 已结束，赛果写入 `stage1_round4_results_2026-06-04.csv`，并合并进 `stage1_round1_4_results_2026-06-05.csv` 推导 `stage1_round4_standings_2026-06-05.csv`。Round 4 决出 GamerLegion、MIBR、M80 三个晋级名额，以及 THUNDER dOWNUNDER、Sharks、HEROIC 三个淘汰名额。

<div align="center">

<img src="docs/images/pickem-round4-checkpoint.svg" width="900" alt="IEM Cologne 2026 Stage 1 Pick'em Round 4 checkpoint and Round 5 watch" />

</div>

Pick'em 进度按三类读：

| 状态 | 队伍 | 含义 |
| --- | --- | --- |
| 🟢 已兑现 | BetBoom、B8、M80、Gaimin Gladiators | 三个 `晋级` 已进 Stage 2；一个 `0-3` 已命中 |
| 🔵 仍可兑现 | BIG、TYLOO | 都是 `晋级` 槽位，Round 5 必须赢 |
| 🔴 槽位已失效 | GamerLegion、MIBR、HEROIC、NRG | GamerLegion/MIBR 已晋级但不是 `3-0`；HEROIC 已淘汰；NRG 已无法 `0-3` |

Round 4 后战绩池：

| 战绩池 | 队伍 | 下一步 |
| --- | --- | --- |
| `3-0 / 已晋级` | BetBoom、B8 | 进入 Stage 2 |
| `3-1 / 已晋级` | GamerLegion、MIBR、M80 | 进入 Stage 2 |
| `2-2` | TYLOO、Lynn Vision、Liquid、FlyQuest、NRG、BIG | Round 5 决胜 BO3，胜者进 Stage 2，负者出局 |
| `1-3 / 已淘汰` | THUNDER dOWNUNDER、Sharks、HEROIC | Stage 1 结束 |
| `0-3 / 已淘汰` | SINNERS、Gaimin Gladiators | Stage 1 结束 |

---

## 🔥 Round 5 关键赛程

以下是 2026-06-05 的 Round 5 fixtures 快照，以 [esports.gg](https://esports.gg/news/counter-strike-2/iem-cologne-major-2026-stage-1-overview-results/) 的 Stage 1 results/schedule 为主，并用 [HLTV Major hub](https://www.hltv.org/major) 与 [bo3.gg 赛程报道](https://bo3.gg/news/big-to-face-nrg-for-a-spot-in-iem-cologne-major-2026-stage-2) 交叉核对。源文件是 `stage1_round5_fixtures_2026-06-05.csv`；合并 Round 4 standings 后的版本是 `stage1_round5_fixtures_with_standings_2026-06-05.csv`。

| 战绩池 | 对阵 | 赛制 | Pick'em 关注点 |
| --- | --- | --- | --- |
| `2-2` 决胜战 | TYLOO vs Lynn Vision | BO3 | TYLOO `晋级` 必须赢；14:00 CEST |
| `2-2` 决胜战 | Liquid vs FlyQuest | BO3 | 非 Pick'em 直接项，但决定最后晋级池；17:00 CEST |
| `2-2` 决胜战 | NRG vs BIG | BO3 | BIG `晋级` 必须赢；NRG `0-3` 已失效；20:00 CEST |

Round 5 的核心看点只剩两个 Pick'em 槽位：TYLOO 和 BIG 都需要赢下 BO3 才能把 `晋级` 槽位补到 5 个以上。Stage 1 全部结束后，再用最终 standings 跑 `backtest-pickem` 计算是否达成 5 分通过线。

> 💡 在 [在线终端](#-静态网站部署) 里点开 Round 5 这几场，模拟不同胜者就能即时看到战绩榜与 Pick'em 槽位如何翻牌。

<details>
<summary>展开 Round 4 逐场结果</summary>

| 轮次 / 战绩池 | 对阵 | 结果 | Pick'em 影响 |
| --- | --- | --- | --- |
| `1-2` 淘汰战 | THUNDER dOWNUNDER vs FlyQuest | FlyQuest 2-1 THUNDER dOWNUNDER | 非 Pick'em 直接项；FlyQuest 进入 Round 5 |
| `1-2` 淘汰战 | TYLOO vs Sharks | TYLOO 2-1 Sharks | TYLOO `晋级` 仍活着，进入 Round 5 |
| `2-1` 晋级战 | GamerLegion vs BIG | GamerLegion 2-0 BIG | GamerLegion 已晋级但 `3-0` 仍错；BIG 掉入 Round 5 |
| `2-1` 晋级战 | MIBR vs Lynn Vision | MIBR 2-1 Lynn Vision | MIBR 已晋级但 `3-0` 仍错；Lynn Vision 掉入 Round 5 |
| `1-2` 淘汰战 | Liquid vs HEROIC | Liquid 2-0 HEROIC | HEROIC `晋级` 已失效；Liquid 进入 Round 5 |
| `2-1` 晋级战 | M80 vs NRG | M80 2-0 NRG | M80 `晋级` 已兑现；NRG `0-3` 仍已失效 |

</details>

<details>
<summary>展开 Day 2 逐场结果（Round 2-3）</summary>

| 轮次 / 战绩池 | 对阵 | 结果 | Pick'em 影响 |
| --- | --- | --- | --- |
| Round 2 `1-0` | B8 vs THUNDER dOWNUNDER | B8 13-11 THUNDER dOWNUNDER | B8 继续保持 `晋级` 路径 |
| Round 2 `0-1` | HEROIC vs Lynn Vision | Lynn Vision 13-11 HEROIC | HEROIC 掉入 `0-2`，晋级容错归零 |
| Round 2 `1-0` | M80 vs Sharks | M80 13-6 Sharks | M80 继续保持 `晋级` 路径 |
| Round 2 `0-1` | MIBR vs TYLOO | MIBR 16-14 TYLOO | MIBR `3-0` 仍已失效；TYLOO 掉入 `0-2` |
| Round 2 `1-0` | GamerLegion vs FlyQuest | GamerLegion 13-11 FlyQuest | GamerLegion 暂时保留 `3-0` 路径 |
| Round 2 `0-1` | SINNERS vs NRG | NRG 13-6 SINNERS | NRG `0-3` 已错 |
| Round 2 `1-0` | BetBoom vs Team Liquid | BetBoom 13-9 Team Liquid | BetBoom 继续保持 `晋级` 路径 |
| Round 2 `0-1` | BIG vs Gaimin Gladiators | BIG 13-1 Gaimin Gladiators | BIG 仍可兑现 `晋级`；Gaimin 进入 `0-2` |
| `2-0` 晋级战 | GamerLegion vs BetBoom | BetBoom 2-0 GamerLegion | BetBoom `晋级` 已兑现；GamerLegion `3-0` 已错 |
| `2-0` 晋级战 | M80 vs B8 | B8 2-0 M80 | B8 `晋级` 已兑现；M80 仍在 `晋级` 路径 |
| `1-1` 调整战 | NRG vs FlyQuest | NRG 13-10 FlyQuest | NRG `0-3` 已错 |
| `1-1` 调整战 | Sharks vs Lynn Vision | Lynn Vision 13-5 Sharks | 非 Pick'em 直接影响 |
| `1-1` 调整战 | Team Liquid vs MIBR | MIBR 13-10 Team Liquid | MIBR `3-0` 已错，但仍可晋级 |
| `1-1` 调整战 | THUNDER dOWNUNDER vs BIG | BIG 13-7 THUNDER dOWNUNDER | BIG `晋级` 仍在路上 |
| `0-2` 淘汰战 | TYLOO vs SINNERS | TYLOO 2-0 SINNERS | TYLOO `晋级` 仍活着；SINNERS 淘汰 |
| `0-2` 淘汰战 | Gaimin Gladiators vs HEROIC | HEROIC 2-0 Gaimin Gladiators | Gaimin Gladiators `0-3` 命中；HEROIC `晋级` 仍活着 |

</details>

---

## 🔬 复盘读数

这里是给玩家看的简版结论；完整命令和诊断字段见 [数据处理与建模二级菜单](docs/data-processing.md#回测与诊断)。

| 层级 | 2026-06-05 快照读数 | 玩家视角结论 |
| --- | --- | --- |
| Day 1 单场 forecast | 有效下注 `3/7`；计入规避方向为 `4/8` | 这版单场模型不能当独立投注信号 |
| 5% margin 重打标 | Day 1 有效 pick 变为 `3/5`；机器推荐可直接输出 `--minimum-margin 0.05` | 低置信 BO1 应该更谨慎 |
| 选手状态信号 | 补入 player-form fixtures 后，5%+player form 的有效 pick 全部带低样本状态风险，命中 `3/5`；叠加状态规避后为 `2/3` | 状态风险能提高命中率但会明显降低覆盖，暂作审查信号 |
| Pick'em 中途状态 | `4 locked / 2 alive / 4 broken` | 提交清单还没终局，至少 4 个槽位已兑现 |
| 槽位分解 | `3-0`: `0/0/2`；`晋级`: `3/2/1`；`0-3`: `1/0/1` | Round 5 只剩 BIG、TYLOO 两个 `晋级` 槽位可补 |
| 候选策略 | `3-0` 建议复查 `extreme_consensus_composite`；`晋级` 和 `0-3` 保持 `status_adjusted_score` | 极端槽位要更重视共识信号，主排序暂不整体替换 |

Stage 1 全部结束前，只做 checkpoint 复盘，不提前计算最终 Pick'em 通过率。

<details>
<summary>🖼️ 赛前预测海报（官方队标，仅用于赛前结果可视化）</summary>

<div align="center">

**3-0**

<img src="docs/images/mibr_3-0.png" width="220" alt="MIBR 3-0 pick" /> <img src="docs/images/gamerlegion_3-0.png" width="220" alt="GamerLegion 3-0 pick" />

**晋级 ADVANCE**

<img src="docs/images/big_advance.png" width="220" alt="BIG advance pick" /> <img src="docs/images/betboom_advance.png" width="220" alt="BetBoom advance pick" /> <img src="docs/images/b8_advance.png" width="220" alt="B8 advance pick" />

<img src="docs/images/heroic_advance.png" width="220" alt="HEROIC advance pick" /> <img src="docs/images/m80_advance.png" width="220" alt="M80 advance pick" /> <img src="docs/images/tyloo_advance.png" width="220" alt="TYLOO advance pick" />

**0-3**

<img src="docs/images/nrg_0-3.png" width="220" alt="NRG 0-3 pick" /> <img src="docs/images/gaimin-gladiators_0-3.png" width="220" alt="Gaimin Gladiators 0-3 pick" />

</div>

</details>

---

## 🚀 静态网站部署

`site/` 是 GitHub Pages 静态站入口，只公开 IEM Cologne Major 2026 的 Swiss 选择器页面：已完成赛果锁定，未完成对局可在浏览器里本地模拟。**纯 ES Module，无需打包工具**——克隆即可起本地服务预览。

```bash
# 1) 重算站点数据（仓库快照数据）
PYTHONPATH=src python3 scripts/update_site_data.py --repo-root . \
  --output-dir data/cologne2026/site_updates --disable-primary --disable-fivee
PYTHONPATH=src python3 scripts/export_site_data.py --repo-root . --output-dir site/data

# 2) 起静态服务（必须 http，fetch 不支持 file://）
python3 -m http.server 8000 --directory site
# 打开 http://localhost:8000
```

前端测试（与 CI 同口径）：

```bash
node --test site/tests/*.test.mjs
```

如需手动生成 AI 分析 JSON，先设置 `AI_UPDATE_NOTES` 和 `AI_API_KEY`，再运行：

```bash
PYTHONPATH=src AI_UPDATE_NOTES="本次只分析最新完赛结果" \
  python3 scripts/generate_ai_articles.py --data-dir site/data --output-dir site/data/ai
```

部署使用 `.github/workflows/pages.yml`：`main` push 会发布静态站；需要更新赛程/分析时，在 GitHub Actions 手动触发 workflow，填写 `ai_update_notes` 后才会生成 AI 内容，并按需把 `refresh_sources` 设为 `true`。AI API key 必须放在 GitHub Secrets 的 `AI_API_KEY`，不要写入仓库。

<details>
<summary>🛠️ 前端技术栈（零构建）</summary>

| 关注点 | 选型 | 说明 |
| --- | --- | --- |
| 框架 | **无** —— 原生 ES Module | `site/src/*.js`，浏览器直接加载，零打包 |
| 字体 | Chakra Petch · Outfit · JetBrains Mono | Google Fonts，含系统字体回退 |
| 动效 | GSAP 3.15（jsDelivr CDN） | 计数入场，`if (window.gsap)` 特性检测，离线降级 |
| 数据可视化 | 手写内联 SVG / CSS | 雷达仪表盘（`r=15.915` 周长=100 技巧）、概率条、投票 pip，零图表库依赖 |
| 氛围背景 | 手写 `<canvas>` 粒子 + CSS 五层 | rAF 循环，`document.hidden` 暂停，reduced-motion 静帧 |
| 图标 | 内联 SVG（lucide 风格） | `currentColor` 继承，离线安全 |
| 可访问性 | `prefers-reduced-motion` 全门控 · `:focus-visible` · ARIA | 动效全部可优雅降级到终态 |

逻辑模块（`swiss.js` / `pickem.js` / `bracket.js`）为纯函数，带单元测试；`render.js` 为纯字符串渲染，DOM 安全。

</details>

---

## 📚 技术二级菜单

| 想看什么 | 入口 |
| --- | --- |
| 数据清洗、特征工程、模型融合、市场信号和 Swiss 模拟链路 | [处理链路](docs/data-processing.md#处理链路) |
| 安装、验证、端到端样例 | [安装与验证](docs/data-processing.md#安装与验证) |
| `update`、`train`、`forecast`、`pickem`、`backtest-*` 等命令 | [命令总览](docs/data-processing.md#命令总览) |
| IEM Cologne 2026 数据文件说明 | [赛事数据资产](docs/data-processing.md#iem-cologne-2026-数据资产) |
| 回测口径、策略重打标、完赛后评分入口 | [回测与诊断](docs/data-processing.md#回测与诊断) |
| CSV 字段、项目结构和模块职责 | [数据输入字段](docs/data-processing.md#数据输入字段) |

---

## 📄 许可

本项目使用 [MIT License](LICENSE)。队标版权归各自战队所有，仅用于赛事结果可视化展示。

<div align="center">
<br>
<sub>// MAJORS.IM · CS2 MAJOR PICK'EM TACTICAL TERMINAL · 融合 专家 0.30 · 市场 0.20 · 模型 0.50</sub>
</div>

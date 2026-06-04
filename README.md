<div align="center">

# CS2 Major Pick'em 助手

<img src="docs/images/readme-hero-v2.png" width="900" alt="CS2 Pick'em prediction banner" />

**给游戏玩家看的 IEM Cologne 2026 Stage 1 Pick'em 清单、赛程进度和复盘**

[数据处理与建模二级菜单](docs/data-processing.md) ·
[赛前答案 JSON](data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/final_fused_pickem_2026-06-01.json) ·
[Round 3 检查点](data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/final_fused_pickem_checkpoint_round3_2026-06-04.json)

</div>

---

## 这页保留什么

README 只保留游戏玩家最关心的信息：赛前 Pick'em 怎么填、2026-06-04 Round 3 后哪些槽位已经兑现、哪些队还活着、Round 4 应该看哪几场。

数据采集、特征工程、模型训练、命令行参数、回测命令、字段说明和模块说明已经移到 [docs/data-processing.md](docs/data-processing.md)。想复现模型或继续处理数据，从那一页开始。

> 仓库快照停在 **2026-06-04 Round 3 后**。Round 4 只保留赛程和压力状态，完整赛果未收齐前不写最终 Pick'em 回测。

## 赛前提交清单

这是 2026-06-01 的赛前最终融合答案单。融合权重为专家 `0.30`、市场 `0.20`、模型 `0.50`，并把选手样本/替补风险写入 `status_adjusted_score`。

| 槽位 | 队伍 | Round 3 后状态 |
| --- | --- | --- |
| `3-0` | MIBR, GamerLegion | 两个 `3-0` 槽位均已失效 |
| `晋级` | BIG, BetBoom, B8, HEROIC, M80, TYLOO | BetBoom、B8 已兑现；BIG、HEROIC、M80、TYLOO 仍可兑现 |
| `0-3` | NRG, Gaimin Gladiators | Gaimin Gladiators 已命中；NRG 已失效 |

> 上表是赛前提交快照，不会因为赛后结果反向修改。

## Round 3 后进度

2026-06-03 Day 2 已结束，Round 2 与 Round 3 共 16 场赛果已经写入 `stage1_day2_results_2026-06-03.csv`，并合并进 `stage1_round1_3_results_2026-06-04.csv` 推导 standings。Round 3 结束后，两个 `2-0` 晋级名额和两个 `0-2` 淘汰名额已经决出。

<div align="center">

<img src="docs/images/pickem-day1-checkpoint.svg" width="900" alt="IEM Cologne 2026 Stage 1 Pick'em Round 3 checkpoint and Round 4 watch" />

</div>

Pick'em 进度按三类读：

| 状态 | 队伍 | 含义 |
| --- | --- | --- |
| 已兑现 | BetBoom、B8、Gaimin Gladiators | 两个 `晋级` 已进 Stage 2；一个 `0-3` 已命中 |
| 仍可兑现 | M80、BIG、TYLOO、HEROIC | 都还在 `晋级` 路径；TYLOO/HEROIC 已经没有容错 |
| 槽位已失效 | GamerLegion、MIBR、NRG | GamerLegion/MIBR 无法再 `3-0`；NRG 已无法 `0-3` |

Round 3 后战绩池：

| 战绩池 | 队伍 | 下一步 |
| --- | --- | --- |
| `3-0 / 已晋级` | BetBoom、B8 | 进入 Stage 2 |
| `2-1` | GamerLegion、M80、NRG、Lynn Vision、MIBR、BIG | Round 4 晋级战，胜者进 Stage 2 |
| `1-2` | FlyQuest、Sharks、Team Liquid、THUNDER dOWNUNDER、TYLOO、HEROIC | Round 4 淘汰战，负者出局 |
| `0-3 / 已淘汰` | SINNERS、Gaimin Gladiators | Stage 1 结束 |

## Round 4 关键赛程

以下是 2026-06-04 的 Round 4 fixtures 快照，以 [esports.gg](https://esports.gg/news/counter-strike-2/iem-cologne-major-2026-stage-1-overview-results/) 的 Stage 1 results/schedule 为主，并用 [HLTV Major hub](https://www.hltv.org/major) 交叉核对。源文件是 `stage1_round4_fixtures_2026-06-04.csv`；合并 Round 3 standings 后的版本是 `stage1_round4_fixtures_with_standings_2026-06-04.csv`。

| 战绩池 | 对阵 | 赛制 | Pick'em 关注点 |
| --- | --- | --- | --- |
| `1-2` 淘汰战 | THUNDER dOWNUNDER vs FlyQuest | BO3 | 非 Pick'em 直接项，但影响最终晋级名额 |
| `1-2` 淘汰战 | TYLOO vs Sharks | BO3 | TYLOO `晋级` 必须连赢两轮，从这里开始 |
| `2-1` 晋级战 | GamerLegion vs BIG | BO3 | BIG 是 `晋级` 关键战；GamerLegion 的 `3-0` 已失效 |
| `2-1` 晋级战 | MIBR vs Lynn Vision | BO3 | MIBR 的 `3-0` 已失效，只影响晋级池竞争 |
| `1-2` 淘汰战 | Team Liquid vs HEROIC | BO3 | HEROIC `晋级` 必须连赢两轮，从这里开始 |
| `2-1` 晋级战 | M80 vs NRG | BO3 | M80 `晋级` 需要赢；NRG `0-3` 已失效 |

Round 4 的核心看点很集中：`晋级` 槽位还剩 M80、BIG、TYLOO、HEROIC 四支活着，其中 M80/BIG 一场 BO3 即可兑现，TYLOO/HEROIC 必须先活过淘汰战再打 Round 5。Round 4 结束后会再确定 3 支晋级队和 3 支淘汰队，剩余 `2-2` 队伍进入 2026-06-05 的 Round 5。

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

## 复盘读数

这里是给玩家看的简版结论；完整命令和诊断字段见 [数据处理与建模二级菜单](docs/data-processing.md#回测与诊断)。

| 层级 | 2026-06-04 快照读数 | 玩家视角结论 |
| --- | --- | --- |
| Day 1 单场 forecast | 有效下注 `3/7`；计入规避方向为 `4/8` | 这版单场模型不能当独立投注信号 |
| 5% margin 重打标 | Day 1 有效 pick 变为 `3/5` | 低置信 BO1 应该更谨慎 |
| 选手状态信号 | 补入 player-form fixtures 后，5%+player form 的有效 pick 全部带低样本状态风险，命中 `3/5`；叠加状态规避后为 `2/3` | 状态风险能提高命中率但会明显降低覆盖，暂作审查信号 |
| Pick'em 中途状态 | `3 locked / 4 alive / 3 broken` | 提交清单还没终局，至少 3 个槽位已兑现 |
| 槽位分解 | `3-0`: `0/0/2`；`晋级`: `2/4/0`；`0-3`: `1/0/1` | 主要损失来自两个 `3-0` 和 NRG `0-3` |
| 候选策略 | `3-0` 建议复查 `extreme_consensus_composite`；`晋级` 和 `0-3` 保持 `status_adjusted_score` | 极端槽位要更重视共识信号，主排序暂不整体替换 |

Stage 1 全部结束前，只做 checkpoint 复盘，不提前计算最终 Pick'em 通过率。

## 赛前预测海报

队标为各战队官方 logo，仅用于赛前结果可视化展示。海报保留原始赛前选择，不随赛果回填。

<div align="center">

**3-0**

<img src="docs/images/mibr_3-0.png" width="240" alt="MIBR 3-0 pick" /> <img src="docs/images/gamerlegion_3-0.png" width="240" alt="GamerLegion 3-0 pick" />

**晋级 ADVANCE**

<img src="docs/images/big_advance.png" width="240" alt="BIG advance pick" /> <img src="docs/images/betboom_advance.png" width="240" alt="BetBoom advance pick" /> <img src="docs/images/b8_advance.png" width="240" alt="B8 advance pick" />

<img src="docs/images/heroic_advance.png" width="240" alt="HEROIC advance pick" /> <img src="docs/images/m80_advance.png" width="240" alt="M80 advance pick" /> <img src="docs/images/tyloo_advance.png" width="240" alt="TYLOO advance pick" />

**0-3**

<img src="docs/images/nrg_0-3.png" width="240" alt="NRG 0-3 pick" /> <img src="docs/images/gaimin-gladiators_0-3.png" width="240" alt="Gaimin Gladiators 0-3 pick" />

</div>

## 技术二级菜单

| 想看什么 | 入口 |
| --- | --- |
| 数据清洗、特征工程、模型融合、市场信号和 Swiss 模拟链路 | [处理链路](docs/data-processing.md#处理链路) |
| 安装、验证、端到端样例 | [安装与验证](docs/data-processing.md#安装与验证) |
| `update`、`train`、`forecast`、`pickem`、`backtest-*` 等命令 | [命令总览](docs/data-processing.md#命令总览) |
| IEM Cologne 2026 数据文件说明 | [赛事数据资产](docs/data-processing.md#iem-cologne-2026-数据资产) |
| 回测口径、策略重打标、完赛后评分入口 | [回测与诊断](docs/data-processing.md#回测与诊断) |
| CSV 字段、项目结构和模块职责 | [数据输入字段](docs/data-processing.md#数据输入字段) |

## 许可

本项目使用 [MIT License](LICENSE)。

<div align="center">

# CS2 Major Pick'em 机器学习预测系统

**离线可运行的 CS2 Major Pick'em 全链路预测工具**
数据清洗 → 特征工程 → 时间切分 → 融合模型 → 瑞士轮蒙特卡洛 → 风险感知 Pick'em 策略

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/core%20deps-zero-success.svg)](#安装)
[![Tests](https://img.shields.io/badge/tests-unittest-brightgreen.svg)](#验证)
[![Status](https://img.shields.io/badge/status-offline--first-orange.svg)](#设计原则)

</div>

---

## 项目简介

`cs2pickem` 是一套面向 CS2 Major **Pick'em 预测**的离线工具链：从原始比赛流水出发，完成数据清洗、无未来泄漏的特征工程、按时间切分的训练/验证/测试、四模型加权融合、瑞士轮蒙特卡洛模拟，最终输出带风险控制的 `3-0 / 晋级 / 0-3` 选择清单。

核心包**不依赖任何第三方库**，用系统 Python 即可端到端运行；安装可选依赖后会自动升级到 scikit-learn / XGBoost / TensorFlow / Matplotlib 等实现，缺失时无缝回退到纯 Python 版本，保证脚本始终可跑。

## 核心特性

- **零依赖即可运行** —— 纯标准库实现核心链路，`ml` / `scrape` / `viz` 可选依赖按需增强。
- **无未来数据泄漏** —— 滚动特征、选手窗口、赔率/BP 合并全部严格按比赛日期截断。
- **四模型加权融合** —— 逻辑回归 + 随机森林 + Boosting 残差树 + 轻量神经网络，默认权重 `20/30/35/15`，并按验证集 Log Loss 赛前微调。
- **瑞士轮蒙特卡洛** —— 蛇形种子配对、同战绩优先、避免复赛，BO3 晋级/淘汰自动处理。
- **风险感知策略** —— 赔率修正、低置信规避、弱队爆冷降权、挑战者/传奇组分层加权。
- **上线门槛审计（readiness）** —— 数据量、字段完整性、模型指标、融合优势、回测通过率、数据源新鲜度一键体检。
- **一键编排（pipeline）** —— 采集后数据 → 训练 → 预测 → 模拟 → 审计 → 最终答案单，全流程串联。

## 处理链路

| 阶段 | 模块 | 说明 |
| --- | --- | --- |
| 1. 数据清洗 | `cleaning` | 过滤二队/Mix/弃赛/低级别赛事、剔除 3σ 异常值、填充 H2H 中性默认值 |
| 2. 特征工程 | `features` `enrichment` | 排名差、RMR 差、Major 历史差、近期胜率、地图胜率、选手 Rating/KD/首杀/残局、赔率、瑞士轮状态等 |
| 3. 时间切分 | `splitting` | 按时间顺序拆分 train/val/test，提供时间序列交叉验证折，带防泄漏日期边界 |
| 4. 模型融合 | `models` `predictor` | 四模型加权融合，记录学习率/树数/深度/叶子样本/子采样/隐藏层超参数 |
| 5. 瑞士轮模拟 | `swiss` `pickem` | 种子配对、同战绩优先、避免复赛，BO3 晋级/淘汰自动处理 |
| 6. Pick'em 策略 | `strategy` `selection` | 输出 `3-0/晋级/0-3` 候选，应用赔率修正、低置信规避、爆冷降权与分组策略 |

## 安装

需要 **Python 3.9+**。免安装即可运行（用 `PYTHONPATH=src`），或安装为可编辑包：

```bash
# 方式一：免安装，直接用源码运行
PYTHONPATH=src python3 -m cs2pickem.cli demo

# 方式二：安装为本地包，获得 `cs2pickem` 命令
pip install -e .
cs2pickem demo
```

可选依赖（按需安装，缺失时自动回退）：

```bash
pip install -e ".[ml]"      # pandas / numpy / scikit-learn / xgboost / tensorflow / joblib / imbalanced-learn
pip install -e ".[scrape]"  # requests / beautifulsoup4（真实抓取）
pip install -e ".[viz]"     # matplotlib（可视化导出）
pip install -e ".[dev]"     # pytest
```

> 下文命令以免安装写法 `PYTHONPATH=src python3 -m cs2pickem.cli <cmd>` 为例；安装后可直接用 `cs2pickem <cmd>`。

## 验证

```bash
# 运行全部单元测试
PYTHONPATH=src python3 -m unittest discover -s tests -v

# 用内置样例跑一次端到端演示
PYTHONPATH=src python3 -m cs2pickem.cli demo
```

## 快速开始

用 `examples/` 里的样例数据跑一次完整离线工作流（采集后的输入 → 训练 → 预测 → 模拟 → 审计 → 答案单）：

```bash
PYTHONPATH=src python3 -m cs2pickem.cli pipeline \
  --history examples/raw_match_history.csv \
  --fixtures examples/upcoming_fixtures.csv \
  --teams examples/sample_teams.csv \
  --odds examples/odds_feed.csv \
  --players examples/player_stats.csv \
  --bp examples/bp_intel.csv \
  --participants examples/major_participants_sample.csv \
  --top-teams examples/top80_teams_sample.csv \
  --version-log examples/version_log.csv \
  --reference-date 2026-05-31 \
  --output-dir /tmp/cs2pickem_pipeline \
  --simulations 100000 --stage challengers --max-age-days 180
```

产物写入 `--output-dir`，包括 `enriched_matches.csv`、`train_report.json`、`forecast_report.json`、`pickem_report.json`、`readiness_report.json`、`pickem_answer_sheet.json` 和 `pipeline_manifest.json`。

## 命令总览

共 22 个子命令，按职责分组：

### 数据采集（本地 HTML 或 `--url` 抓取）

| 命令 | 作用 |
| --- | --- |
| `update` | 抓取/解析 HLTV-like 结果页 → JSON 数据集 + manifest，可增量追加长期训练 CSV |
| `daily-update` | 配置驱动的多源每日增量更新，统一去重追加 + per-job/总 manifest（适配 cron/launchd） |
| `event-teams` | 解析 HLTV-like event 页 → 参赛队伍/种子/排名/晋级来源 CSV |
| `rankings` | 解析 HLTV-like ranking 页 → Top-N 队伍/积分/区域 CSV（默认 `--limit 80`） |
| `player-stats` | 解析 HLTV-like 选手统计页 → Rating/KD/首杀/残局/替补 CSV |
| `fivee-collect` | 低频抓取 5E 战队页 → `fivee_teams/players/maps.csv` + manifest |
| `fivee-match-results` | 反向翻页抓取 5E 全局赛果接口 → 指定日期窗口赛果 CSV |

### 数据处理 / 特征

| 命令 | 作用 |
| --- | --- |
| `enrich` | 原始流水 → 无泄漏滚动特征（近 5/10 场胜率、30 天参赛量、BO 胜率、连胜连败、H2H）+ 队伍地图画像 |
| `merge-odds` | 多平台十进制赔率按日期+无序队伍对齐合并，输出均值赔率与归一化市场概率 |
| `merge-players` | 只读赛前窗口内选手行，合并 Rating/KD/首杀/残局/明星/替补，避免未来泄漏 |
| `merge-bp` | 按日期+无序队伍合并赛前地图 BP 情报（确认地图、双方禁/选图、来源、置信度） |

### 建模 / 评估

| 命令 | 作用 |
| --- | --- |
| `train` | 清洗 → 切分 → 四模型融合训练 → 输出 Accuracy/AUC/LogLoss/盈亏、CV、BO1/BO3 分段、单模型 vs 融合对比 |
| `visualize` | 训练报告 → 特征重要性图 + 预测概率分布图（Matplotlib PNG，回退 SVG） |

### 预测 / 策略

| 命令 | 作用 |
| --- | --- |
| `forecast` | 赛前 fixtures 单场胜率：四模型独立胜率 + 加权贡献 + ±3% 赔率修正 + 低置信规避 + 候选地图分布 |
| `pickem` | 用融合模型作为 Swiss 胜率函数 → 蒙特卡洛 → `3-0/晋级/0-3` 清单 + 逐队风险拆解 |
| `simulate` | 输出每队 `3-0/3-1/3-2/0-3/1-3/2-3/晋级/淘汰` 概率与 Pick'em 候选 |
| `answer-sheet` | 把大型 Pick'em + readiness 报告压缩为可提交/复核的最终答案单 |

### 编排 / 审计 / 回测

| 命令 | 作用 |
| --- | --- |
| `pipeline` | 串联 enrich→增强→merge→train→visualize→forecast→pickem→readiness 的一键离线工作流 |
| `readiness` | 上线门槛审计：数据量 ≥8000、字段完整性、名单覆盖、模型指标、融合优势、回测通过率、数据源新鲜度等 |
| `demo` | 用内置样例跑一遍核心链路演示 |
| `backtest-pickem` | Pick'em 报告 vs 最终 Swiss standings，计算命中数与是否达 pass threshold |
| `backtest-pickem-suite` | 多场 Major 的 suite 级通过率汇总（默认目标 38%） |

> 每个命令的完整参数见 `cs2pickem <cmd> --help`。

## 端到端示例

从原始数据到最终答案单的关键步骤（更多组合见 `cs2pickem --help`）：

```bash
# 1) 生成滚动特征与地图画像
PYTHONPATH=src python3 -m cs2pickem.cli enrich \
  --matches examples/raw_match_history.csv \
  --output /tmp/enriched.csv --profiles-output /tmp/profiles.json

# 2) 训练融合模型（六个月窗口传 --max-age-days 180）
PYTHONPATH=src python3 -m cs2pickem.cli train \
  --matches examples/sample_matches.csv --reference-date 2026-05-31 \
  --top-k 25 --cv-folds 5 --max-age-days 180 --output /tmp/train_report.json

# 3) 赛前 fixtures 单场预测
PYTHONPATH=src python3 -m cs2pickem.cli forecast \
  --history /tmp/enriched.csv --fixtures examples/upcoming_fixtures.csv \
  --profiles /tmp/profiles.json --reference-date 2026-05-31 --top-k 25 --max-age-days 180

# 4) 瑞士轮蒙特卡洛 + Pick'em 清单
PYTHONPATH=src python3 -m cs2pickem.cli pickem \
  --history /tmp/enriched.csv --teams examples/sample_teams.csv \
  --fixtures examples/upcoming_fixtures.csv --profiles /tmp/profiles.json \
  --reference-date 2026-05-31 --simulations 100000 --stage challengers --max-age-days 180
```

### 关于训练窗口与门槛

- 默认清洗窗口 90 天；赛前六个月训练传 `--max-age-days 180`。
- 严格复现目标日历切分（2026.01–2026.04 / 2026.05 上旬 / 2026.05 下旬）：加 `--train-end-date 2026-04-30 --validation-end-date 2026-05-15`。显式拆分要求 train/val/test 三段都有数据，避免被错误边界切成伪评估。
- 生产级 `readiness` 会强制校验：日历拆分、采集行级范围、验证集调权、50%–52% 低置信单场全部 `avoid`、≥10 万次模拟、完整 `3-0/晋级/0-3` 槽位、每个入选项 ≥4% 选择边际、至少 1 个 Swiss matchup 使用真实赔率、关键数据源不超过赛前窗口快照。

## 数据输入字段

CSV 或字典行建议包含以下列；`readiness` 的字段完整性门槛会要求核心建模列已填充：

- **基础**：`date` `event` `event_tier` `status` `team1` `team2` `winner` `best_of` `map`
- **队伍**：`team{1,2}_rank` `team{1,2}_rmr_points` `team{1,2}_major_best_placement`
- **近期状态**：`team{1,2}_matches_30d` `team{1,2}_recent_winrate_{5,10}` `team{1,2}_bo{1,3}_winrate_6m` `team{1,2}_current_streak`
- **地图**：`team{1,2}_map_winrate`
- **选手**：`team{1,2}_rating` `team{1,2}_kd` `team{1,2}_opening_success` `team{1,2}_clutch_winrate` `team{1,2}_star_rating` `team{1,2}_substitute_flag` `team{1,2}_player_sample`
- **对局**：`h2h_team1_winrate` `odds_team1` `odds_team2` `swiss_round` `team{1,2}_wins` `team{1,2}_losses` `version_tag`
- **BP 情报**：`date` `source` `team1` `team2` `map`/`confirmed_map`/`expected_map` `confidence` `team{1,2}_bans` `team{1,2}_pick`

## 项目结构

```
CS-AI-bet/
├── src/cs2pickem/            # 核心 Python 包（零三方依赖即可运行）
│   ├── cli.py                # 命令行入口（22 个子命令）
│   ├── pipeline.py / workflow.py   # 一键离线工作流编排
│   ├── cleaning.py           # 数据清洗
│   ├── enrichment.py         # 无泄漏滚动特征 + 队伍地图画像
│   ├── features.py / selection.py / imbalance.py
│   ├── models.py / predictor.py / evaluation.py / splitting.py
│   ├── swiss.py / pickem.py / strategy.py / forecast.py
│   ├── sources.py / update.py / fivee.py / dataset_store.py
│   ├── odds.py / players.py / bp.py / maps.py / ratings.py
│   └── readiness.py / backtest.py / export.py / visualization.py
├── tests/                    # 单元测试（标准库 unittest）
├── examples/                 # 样例 CSV/HTML/JSON 输入
├── data/                     # IEM Cologne 2026 真实数据与预测产物
│   ├── cologne2026/          # raw / processed / manifests / predictions
│   └── fivee/                # 5EPlay 抓取数据
├── docs/                     # 实施计划等文档
├── pyproject.toml
└── README.md
```

## 模块说明

| 模块 | 职责 |
| --- | --- |
| `sources` / `update` | HTTP 缓存、HLTV-like 结果/event/ranking/选手页解析、版本标签、数据集 manifest |
| `daily-update` | 配置驱动的多源每日增量更新，去重追加长期训练 CSV |
| `dataset_store` | 长期训练 CSV 增量追加、去重、覆盖范围 manifest |
| `enrichment` | 从原始流水生成无未来泄漏的滚动状态特征与队伍地图画像 |
| `forecast` | 赛前 fixtures 单场胜率、赔率修正、低置信规避、未知地图 Top3 均值预测 |
| `bp` | 合并赛前地图 BP 情报，确认地图后改用确认地图特征 |
| `odds` | 多平台赔率归一化、反向队伍匹配、均值合并与覆盖率报告 |
| `players` | 按赛前 lookback 窗口把选手统计聚合成队伍级特征 |
| `readiness` | 上线前数据量、字段完整性、模型指标与融合优势审计 |
| `selection` | 低方差过滤、Pearson 相关冗余过滤、按标签相关性保留 TOP-K 特征 |
| `imbalance` | 确定性 SMOTE-like 上采样与类别权重，训练/CV/预测共用 |
| `maps` | 未知 BP 时按双方 ban/pick 偏好与地图胜率生成 Top3 候选并取均值 |
| `visualization` | 从训练报告导出特征重要性图、预测概率分布图与 manifest |
| `export` | 从 Pick'em/readiness 报告生成最终答案单、选择边际与警告摘要 |
| `swiss` / `pickem` / `strategy` | 瑞士轮模拟、Pick'em 清单生成与风险感知选择策略 |
| `workflow` / `pipeline` | 一键离线编排训练、预测、模拟、审计与答案单输出 |

## IEM Cologne 2026 数据

`data/cologne2026/` 收录了 2026-06-01 核对的真实赛事数据与预测产物（raw 抓取、processed 特征、manifests、predictions）；`data/fivee/` 是对应的 5EPlay 抓取数据。

- `examples/cologne2026_participants.csv` —— 2026-06-01 核对的 32 队全量参赛快照，含 Stage 1/2/3 起始阶段与 VRS/RMR 分数。
- `examples/cologne2026_stage1_teams.csv` / `examples/cologne2026_stage1_opening_fixtures.csv` —— Stage 1 专用输入快照，用于验证真实队名/首轮 fixtures 的管线兼容性。

> 这些是赛前快照，不是长期训练数据，也不替代赛前最新抓取、赔率与选手状态更新。

## 设计原则

- **离线优先**：核心链路不伪造实时 HLTV 结果、赔率或选手状态；赛前需先采集干净数据再训练/模拟。
- **可复现**：所有切分、采样、模拟均确定性可控，报告保留超参数与边界，便于审计回放。
- **优雅降级**：可选依赖缺失时自动回退纯 Python 实现，脚本永远可跑。

## 许可

暂未声明开源许可（默认保留所有权利）。如需开放使用，请在仓库中补充 `LICENSE` 文件。

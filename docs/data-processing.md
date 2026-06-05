# 数据处理与建模二级菜单

本页面向需要复现、审计或继续开发 `cs2pickem` 的读者。游戏玩家只看赛前答案、赛程进度和简版复盘时，回到 [README](../README.md) 即可。

## 处理链路

<p align="center">
  <img src="images/pipeline-overview.png" width="900" alt="CS2 Pick'em offline prediction pipeline visual" />
</p>

`cs2pickem` 是一套面向 CS2 Major Pick'em 的离线工具链：从原始比赛流水出发，完成数据清洗、无未来泄漏的滚动特征、在线赛前 Elo、按时间切分的训练/验证/测试、验证集概率校准、模型融合、市场赔率/民调信号审计、瑞士轮蒙特卡洛模拟，最终输出带风险控制的 `3-0 / 晋级 / 0-3` 选择清单。

核心包不依赖第三方库，用系统 Python 即可端到端运行；安装可选依赖后会优先使用 scikit-learn / XGBoost / joblib 加速逻辑回归、随机森林与 XGBoost 分量。任一加速依赖缺失或导入失败时，会自动回退到纯 Python 实现。神经网络分量默认保留纯 Python 路径且默认权重为 0，只有显式设置 `CS2PICKEM_ACCELERATED_MLP=1` 才尝试 sklearn MLP。

| 阶段 | 模块 | 说明 |
| --- | --- | --- |
| 1. 数据清洗 | `cleaning` | 过滤二队/Mix/弃赛/低级别赛事、剔除 3σ 异常值、填充 H2H 中性默认值 |
| 2. 特征工程 | `features` `enrichment` `reliability` | 排名差、RMR 差、Major 历史差、近期胜率、地图胜率、选手状态、赛前 Elo、瑞士轮状态等 |
| 3. 时间切分 | `splitting` | 按时间顺序拆分 train/val/test，提供时间序列交叉验证折，带防泄漏日期边界 |
| 4. 模型融合 | `models` `predictor` | Logistic / RandomForest / XGBoost 主栈，自动记录后端、权重、超参数与特征准备策略 |
| 5. 校准调参 | `calibration` `tuning` | 验证集 Platt 校准、滚动 fold 评估、候选模型/Top-K/Elo/市场权重对比 |
| 6. 市场信号 | `odds` `forecast` `pickem` | 十进制赔率、5E 别名、美式赔率、显式市场概率、HLTV poll proxy 的统一解析与审计 |
| 7. Pick'em 策略 | `swiss` `strategy` `selection` | 瑞士轮模拟并输出 `3-0/晋级/0-3` 候选，应用低置信规避和风险分层 |

核心特性：

- 零依赖即可运行：核心链路只用标准库，`ml` / `scrape` / `viz` 可选依赖按需增强。
- 无未来数据泄漏：滚动特征、赛前 Elo、选手窗口、赔率/BP 合并全部按比赛日期截断，并默认剔除不稳定身份特征。
- 三模型主栈 + 可审计融合：默认主栈为 LogisticRegression / RandomForest / XGBoost，原始权重 `0.20/0.30/0.35/0.00`，模型内部归一化；纯 Python NN 作为保底组件保留。
- 概率校准与回归验证：训练报告与 `optimize-matches` 支持按时间切分、滚动验证、Brier/ECE/Log Loss、Platt 校准、with/without Elo 对比。
- 市场信号有边界：真实赔率会参与轻量修正；HLTV fan poll 等民调只作为 proxy 报告，不直接当赔率使用。
- 瑞士轮蒙特卡洛：蛇形种子配对、同战绩优先、避免复赛，BO3 晋级/淘汰自动处理。
- 风险感知策略：赔率修正、低置信规避、弱队爆冷降权、挑战者/传奇组分层加权。
- 上线门槛审计：数据量、字段完整性、模型指标、融合优势、回测通过率、数据源新鲜度一键体检。
- 一键编排：采集后数据 → 训练 → 预测 → 模拟 → 审计 → 最终答案单，全流程串联。

## 安装与验证

需要 Python 3.9+。免安装即可运行（用 `PYTHONPATH=src`），或安装为可编辑包：

```bash
# 方式一：免安装，直接用源码运行
PYTHONPATH=src python3 -m cs2pickem.cli demo

# 方式二：安装为本地包，获得 cs2pickem 命令
pip install -e .
cs2pickem demo
```

可选依赖按需安装，缺失时自动回退：

```bash
pip install -e ".[ml]"      # pandas / numpy / scikit-learn / xgboost / joblib 等 ML 加速依赖
pip install -e ".[scrape]"  # requests / beautifulsoup4（真实抓取）
pip install -e ".[viz]"     # matplotlib（可视化导出）
pip install -e ".[dev]"     # pytest
```

> `.[ml]` 中仍声明了实验性依赖（如 TensorFlow / imbalanced-learn），但默认模型路径不依赖 TensorFlow。推荐的加速后端是 scikit-learn + XGBoost + joblib。

验证命令：

```bash
# 运行全部单元测试
PYTHONPATH=src python3 -m unittest discover -s tests -v

# 用内置样例跑一次端到端演示
PYTHONPATH=src python3 -m cs2pickem.cli demo

# 检查模型实际使用的后端
PYTHONPATH=src python3 - <<'PY'
from cs2pickem.models import default_ensemble
print(default_ensemble(seed=7, epochs=2, n_jobs=1).component_backends)
PY
```

安装推荐加速依赖后，后端检查通常应类似：

```text
{'logistic': 'sklearn', 'random_forest': 'sklearn', 'xgboost': 'xgboost', 'neural_network': 'pure_python'}
```

## 快速开始

用 `examples/` 里的样例数据跑一次完整离线工作流：

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

每个命令的完整参数见 `PYTHONPATH=src python3 -m cs2pickem.cli <cmd> --help`，安装后可直接用 `cs2pickem <cmd> --help`。

### 数据采集

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
| `merge-odds` | 合并十进制赔率、5E 别名或美式赔率；优先按 `source_match_url`，否则按日期+无序队伍匹配，输出市场概率与匹配审计 |
| `merge-players` | 只读赛前窗口内选手行，合并 Rating/KD/首杀/残局/明星/替补，并生成短期 player form，避免未来泄漏 |
| `merge-bp` | 按日期+无序队伍合并赛前地图 BP 情报（确认地图、双方禁/选图、来源、置信度） |
| `merge-standings` | 把 Swiss standings 写回下一轮 fixtures，补齐 `team*_record` 与 `swiss_match_type`，避免中途预测沿用赛前 0-0 状态 |

### 建模 / 评估

| 命令 | 作用 |
| --- | --- |
| `train` | 清洗 → 切分 → 融合训练 → 输出 Accuracy/AUC/LogLoss/盈亏、CV、BO1/BO3 分段、单模型 vs 融合对比 |
| `optimize-matches` | 回放历史比赛，比较模型候选、Top-K 特征、Elo 开关、校准和市场融合权重，输出验证/测试/滚动评估 |
| `visualize` | 训练报告 → 特征重要性图 + 预测概率分布图（Matplotlib PNG，回退 SVG） |

### 预测 / 策略

| 命令 | 作用 |
| --- | --- |
| `forecast` | 赛前 fixtures 单场胜率：模型分量胜率 + 加权贡献 + 市场信号修正 + 选手状态摘要 + 可配置低置信/BO1/选手状态规避 + 候选地图分布 |
| `apply-forecast-policy` | 不重训模型，直接对既有 forecast 报告应用 minimum margin、BO1 margin 和 player form 规避策略，适合赛后快速调参复盘 |
| `pickem` | 用融合模型作为 Swiss 胜率函数 → 蒙特卡洛 → `3-0/晋级/0-3` 清单 + 逐队风险拆解 |
| `simulate` | 输出每队 `3-0/3-1/3-2/0-3/1-3/2-3/晋级/淘汰` 概率与 Pick'em 候选 |
| `answer-sheet` | 把大型 Pick'em + readiness 报告压缩为可提交/复核的最终答案单 |

### 编排 / 审计 / 回测

| 命令 | 作用 |
| --- | --- |
| `pipeline` | 串联 enrich→增强→merge→train→visualize→forecast→pickem→readiness 的一键离线工作流 |
| `readiness` | 上线门槛审计：数据量 ≥8000、字段完整性、名单覆盖、模型指标、融合优势、回测通过率、数据源新鲜度等 |
| `demo` | 用内置样例跑一遍核心链路演示 |
| `backtest-forecast` | 单场 forecast 报告 vs 实际赛果 CSV，输出有效下注命中率、方向命中率、低置信规避、BO1 专用阈值候选、市场修正、favorite upset、player form 和 Swiss 压力诊断 |
| `standings-from-results` | 从逐场赛果 CSV 自动推导 Swiss `team,wins,losses,status` standings，减少手写战绩表错误 |
| `backtest-pickem` | Pick'em 报告 vs 最终 Swiss standings，计算命中数与是否达 pass threshold |
| `checkpoint-pickem` | Pick'em 报告 vs Swiss standings 快照，输出每个槽位 locked / alive / broken、下一场锁定/破损压力、状态/槽位 confidence 诊断 |
| `backtest-pickem-suite` | 多场 Major 的 suite 级通过率汇总（默认目标 38%） |
| `replay-pickem-suite` | 重新训练、生成并评分历史 Pick'em replay cases，避免只测静态旧报告 |

## 端到端示例

从原始数据到最终答案单的关键步骤：

```bash
# 1) 生成滚动特征与地图画像
PYTHONPATH=src python3 -m cs2pickem.cli enrich \
  --matches examples/raw_match_history.csv \
  --output /tmp/enriched.csv --profiles-output /tmp/profiles.json

# 2) 训练融合模型（六个月窗口传 --max-age-days 180）
PYTHONPATH=src python3 -m cs2pickem.cli train \
  --matches examples/sample_matches.csv --reference-date 2026-05-31 \
  --top-k 25 --cv-folds 5 --max-age-days 180 --output /tmp/train_report.json

# 3) 回归验证候选逻辑，重点看验证集/测试集/滚动 fold
PYTHONPATH=src python3 -m cs2pickem.cli optimize-matches \
  --matches /tmp/enriched.csv --reference-date 2026-05-31 \
  --max-age-days 180 --top-k-values 12,18,25 \
  --candidates fast_logistic,random_forest,no_nn \
  --elo-modes with,without --output /tmp/match_tuning_report.json

# 4) 赛前 fixtures 单场预测
PYTHONPATH=src python3 -m cs2pickem.cli forecast \
  --history /tmp/enriched.csv --fixtures examples/upcoming_fixtures.csv \
  --profiles /tmp/profiles.json --reference-date 2026-05-31 --top-k 25 --max-age-days 180 \
  --minimum-margin 0.05 --avoid-player-form-counter-signal

# 5) 瑞士轮蒙特卡洛 + Pick'em 清单
PYTHONPATH=src python3 -m cs2pickem.cli pickem \
  --history /tmp/enriched.csv --teams examples/sample_teams.csv \
  --fixtures examples/upcoming_fixtures.csv --profiles /tmp/profiles.json \
  --reference-date 2026-05-31 --simulations 100000 --stage challengers --max-age-days 180
```

## 合并市场赔率

`merge-odds` 接受三类真实赔率输入：`odds_team1/odds_team2`、5E 常见别名 `team1_odds/team2_odds`、美式赔率 `odds_team1_american/odds_team2_american` 等。合并后会写入 `market_probability_team1`、`market_signal_basis`、`market_signal_source`、`market_signal_proxy`，供 `forecast` / `pickem` 审计和轻量修正。

```bash
# 历史比赛合并 5E 赔率
PYTHONPATH=src python3 -m cs2pickem.cli merge-odds \
  --matches data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/enriched_matches.csv \
  --odds data/cologne2026/processed/fivee_stage1_match_results_6m_2026-06-01.csv \
  --output data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/enriched_matches_with_5e_odds.csv

# 赛前 fixtures 合并开盘美式赔率
PYTHONPATH=src python3 -m cs2pickem.cli merge-odds \
  --matches data/cologne2026/processed/stage1_opening_fixtures_fivee_6m_model_2026-06-01.csv \
  --odds data/cologne2026/source_inputs/opening_match_odds_american_2026-06-01.csv \
  --output data/cologne2026/processed/stage1_opening_fixtures_fivee_6m_model_with_market_odds_2026-06-01.csv
```

民调代理字段 `hltv_poll_team1/hltv_poll_team2` 也会被解析为 `poll_proxy`，但只用于报告，不用于真实赔率修正。

## 训练窗口与门槛

- 默认清洗窗口 90 天；赛前六个月训练传 `--max-age-days 180`。
- 严格复现目标日历切分时，加 `--train-end-date 2026-04-30 --validation-end-date 2026-05-15`。显式拆分要求 train/val/test 三段都有数据，避免被错误边界切成伪评估。
- 生产级 `readiness` 会强制校验：日历拆分、采集行级范围、验证集调权、50%-52% 低置信单场全部 `avoid`、≥10 万次模拟、完整 `3-0/晋级/0-3` 槽位、每个入选项 ≥4% 选择边际、至少 1 个 Swiss matchup 使用真实赔率、关键数据源不超过赛前窗口快照。
- 如果要强制确认选手状态已经进入训练层，而不是只存在于赛前 fixtures 或赛后风险报告里，给 `readiness` 加 `--minimum-player-status-features N`。它会读取 `training_report.feature_selection.required_features.selected`，低于 N 时把 `player_status_features` 标为失败。当前 2026-06-04 logistic-only 诊断中 selected 为 0，说明六个月历史训练集还没有可学习的状态特征方差。

## IEM Cologne 2026 数据资产

`data/cologne2026/` 收录了 2026-06-01 核对的真实赛事数据与预测产物（raw 抓取、processed 特征、manifests、predictions、source_inputs）；`data/fivee/` 是对应的 5EPlay 抓取数据。

| 路径 | 说明 |
| --- | --- |
| `examples/cologne2026_participants.csv` | 2026-06-01 核对的 32 队全量参赛快照，含 Stage 1/2/3 起始阶段与 VRS/RMR 分数 |
| `examples/cologne2026_stage1_teams.csv` / `examples/cologne2026_stage1_opening_fixtures.csv` | Stage 1 专用输入快照，用于验证真实队名/首轮 fixtures 的管线兼容性 |
| `data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/enriched_matches_with_5e_odds.csv` | 6 个月历史训练集已合并 5E 真实赔率 |
| `data/cologne2026/processed/stage1_opening_fixtures_fivee_6m_model_with_market_odds_2026-06-01.csv` | Stage 1 首轮 fixtures 已合并开盘美式赔率 |
| `forecast_report.json` / `pickem_report.json` / `pickem_answer_sheet.json` | 当前主报告使用真实市场赔率版本 |
| `stage1_day1_results_2026-06-02.csv` / `forecast_backtest_day1_2026-06-02.json` | Day 1 首轮真实赛果和 `backtest-forecast` 机器回测产物 |
| `forecast_status_required_logistic_2026-06-04.json` / `forecast_status_required_logistic_backtest_day1_2026-06-02.json` | 状态特征 required-selection 的 logistic-only 快速诊断；用于确认训练历史是否真的提供可学习的选手状态信号，不替代赛前主报告 |
| `stage1_day2_results_2026-06-03.csv` | Day 2 的 Round 2-3 共 16 场赛果单独归档 |
| `stage1_round1_3_results_2026-06-04.csv` / `stage1_round3_standings_2026-06-04.csv` / `final_fused_pickem_checkpoint_round3_2026-06-04.json` | Round 1-3 已复核赛果、Round 3 standings 和 `checkpoint-pickem` 中途状态报告 |
| `stage1_round4_fixtures_2026-06-04.csv` / `stage1_round4_results_2026-06-04.csv` | Round 4 赛程快照和六场 BO3 赛果 |
| `stage1_round1_4_results_2026-06-05.csv` / `stage1_round4_standings_2026-06-05.csv` / `final_fused_pickem_checkpoint_round4_2026-06-05.json` | Round 1-4 已复核赛果、Round 4 standings 和 `checkpoint-pickem` 中途状态报告 |
| `stage1_round5_fixtures_2026-06-05.csv` / `stage1_round5_fixtures_with_standings_2026-06-05.csv` | Round 5 三场 2-2 决胜赛程，以及合并 Round 4 standings 后的晋级/淘汰压力 fixtures |
| `final_fused_pickem_2026-06-01.json` / `final_fused_pickem_table_2026-06-01.csv` | 专家/市场/模型最终融合结果；保留 `raw_fused_score / player_availability_multiplier / status_adjusted_score` 和候选池 scoreboard |
| `forecast_without_market_odds_2026-06-01.json` / `pickem_without_market_odds_2026-06-01.json` | 无市场赔率备份，便于对比 |

## 静态网站数据导出

GitHub Pages 站点只读取 `site/data/*.json`。用下面命令从已复核的赛事数据生成静态 JSON：

```bash
PYTHONPATH=src python3 scripts/update_site_data.py \
  --repo-root . \
  --output-dir data/cologne2026/site_updates \
  --disable-primary \
  --disable-fivee

PYTHONPATH=src python3 scripts/export_site_data.py \
  --repo-root . \
  --output-dir site/data

PYTHONPATH=src python3 scripts/generate_ai_articles.py \
  --data-dir site/data \
  --output-dir site/data/ai
```

`update_site_data.py` 在 GitHub Actions 定时任务中会启用主来源和 5E fallback；成功生成的 `data/cologne2026/site_updates` 与 `site/data` 会提交回仓库，作为下一次失败时的有效缓存。本地文档命令使用 `--disable-primary --disable-fivee` 方便离线验证。`generate_ai_articles.py` 在 `AI_API_KEY` 存在时调用 OpenAI-compatible API；没有 key 或 API 失败时写入模板 fallback 文章。

## 回测与诊断

回测拆成三层，避免把“单场胜负预测”“Pick'em 槽位中途状态”和“最终 Pick'em 命中”混在一起。

<p align="center">
  <img src="images/backtest-diagnostics.png" width="900" alt="CS2 Pick'em backtest diagnostics dashboard visual" />
</p>

| 层级 | 评估对象 | 2026-06-05 读数 | 结论用途 |
| --- | --- | --- | --- |
| 单场 forecast 回测 | Day 1 首轮 8 场 BO1；`avoid` 不计入有效下注 | 有效下注 `3/7`；计入规避方向为 `4/8`；已写入 `forecast_backtest_day1_2026-06-02.json` | 诊断单场模型、市场修正、低置信规避和 player form 是否合理 |
| Pick'em 槽位中途回测 | 赛前 `3-0 / 晋级 / 0-3` 槽位对照 Round 4 结束后的战绩 | `checkpoint-pickem` 输出 `4 locked / 2 alive / 4 broken`；已写入 `final_fused_pickem_checkpoint_round4_2026-06-05.json` | 追踪提交清单的兑现路径，但不提前计算最终通过率 |
| 最终 Pick'em 回测 | Stage 1 完赛后的最终 Swiss standings | 待 Stage 1 结束后用 `backtest-pickem` 计算 | 判断是否达到 pass threshold，并进入 readiness 审计 |

当前策略诊断结论：

- Day 1 首轮有效下注只命中 `3/7`，说明这版单场模型尚不能作为独立投注信号；`B8 vs TYLOO` 的低置信规避虽然方向偏对，但正确避免把 50.9% 当成强信号。
- 把赛前单场 minimum margin 提到 `0.05` 后，Day 1 有效 pick 变成 `3/5`；`policy_tradeoff_summary.recommended_policy_update` 会把原始 Day 1 回测的候选直接翻译成 `apply-forecast-policy` 参数和 CLI flags，例如 `--minimum-margin 0.05`。`--bo1-minimum-margin 0.05` 可只收紧 BO1，不把 BO3 一起收紧。
- `market favorite ≥0.60 且 player form 反向则 avoid` 能提高百分比但覆盖太低，只作为低覆盖候选，不作为默认策略。
- `player_status_signal_risk` 会把实际赛果按被选中一侧的低样本/替补风险拆开。原始赛前 `forecast_report.json` 没有 player status 字段；补入 player-form fixtures 后，5%+player form 的 5 个有效 pick 全部带低样本状态风险，命中 `3/5`，状态规避版本降到 3 个有效 pick，命中 `2/3`。因此 `--avoid-player-status-risk` 只作为审查信号，不替换 5%+player form 默认策略。
- 模型训练层新增 `feature_selection.required_features` 诊断，会把选手 rating/KD/首杀/残局/star、替补、样本量、player form 和样本置信度列为 required 候选。2026-06-04 的 logistic-only 快速诊断显示，当前 6 个月训练历史中这些状态特征全部 `unavailable`，Day 1 回测为 `4/8`；这说明现阶段缺的是无泄漏历史选手状态数据，不能把 fixtures 上的状态字段当作模型已经学到的信号。
- Pick'em 层面，BetBoom、B8、M80 晋级和 Gaimin Gladiators `0-3` 已经兑现，BIG/TYLOO 仍能在 Round 5 补回晋级槽；GamerLegion/MIBR 的 `3-0`、HEROIC 的 `晋级` 与 NRG 的 `0-3` 已经不可恢复。
- `candidate_scoreboard_policy_diagnostics` 把 `3-0` 标为 `review_candidate_policy / extreme_consensus_composite`，`advance` 和 `0-3` 继续 `keep_current_policy / status_adjusted_score`。

### Day 1 forecast 回测

```bash
PYTHONPATH=src python3 -m cs2pickem.cli backtest-forecast \
  --forecast-report data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/forecast_report.json \
  --results data/cologne2026/source_inputs/stage1_day1_results_2026-06-02.csv \
  --output data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/forecast_backtest_day1_2026-06-02.json
```

`backtest-forecast` 会按日期 + 无序队伍匹配赛果，逐场输出 pick、directional pick、实际 winner、比分、地图、低置信规避、市场修正、favorite/model/market favorite、player form 分差、被选中一侧的 player status、Swiss `swiss_match_type` 压力类型、`avoid_reason_diagnostics`、`swiss_pressure_diagnostics`、`bo1_margin_policy_candidates`、`player_status_signal_risk`、`player_status_policy_candidates`、`policy_tradeoff_summary`，以及赛后 minimum-margin 阈值候选曲线。forecast / pickem / train 报告会输出 `feature_selection.required_features`，用于审计 required 选手状态特征是 `available`、`selected` 还是因为历史数据无方差而 `unavailable`。`policy_tradeoff_summary.recommended_policy_update` 会额外给出 `apply_forecast_policy_args` 和 `cli_flags`；如果结论是 `keep_current_policy`，也会回显原 forecast report 的 `decision_policy`，便于把实际赛果里验证过的当前策略复用到下一轮 forecast。

### 不重训策略重打标

```bash
PYTHONPATH=src python3 -m cs2pickem.cli apply-forecast-policy \
  --forecast-report data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/forecast_report.json \
  --fixtures data/cologne2026/processed/stage1_opening_fixtures_fivee_6m_model_with_market_odds_player_form_2026-06-04.csv \
  --minimum-margin 0.05 --avoid-player-form-counter-signal \
  --player-form-counter-min-confidence 0.4 \
  --output data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/forecast_policy_margin_0_05_player_form_2026-06-04.json
```

只提高 BO1 门槛时：

```bash
PYTHONPATH=src python3 -m cs2pickem.cli apply-forecast-policy \
  --forecast-report data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/forecast_report.json \
  --fixtures data/cologne2026/processed/stage1_opening_fixtures_fivee_6m_model_with_market_odds_player_form_2026-06-04.csv \
  --minimum-margin 0.02 --bo1-minimum-margin 0.05 \
  --avoid-player-form-counter-signal \
  --player-form-counter-min-confidence 0.4 \
  --output data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/forecast_policy_bo1_margin_0_05_player_form_2026-06-04.json
```

高精度/低覆盖候选：

```bash
PYTHONPATH=src python3 -m cs2pickem.cli apply-forecast-policy \
  --forecast-report data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/forecast_report.json \
  --fixtures data/cologne2026/processed/stage1_opening_fixtures_fivee_6m_model_with_market_odds_player_form_2026-06-04.csv \
  --minimum-margin 0.05 --avoid-player-form-counter-signal \
  --player-form-counter-min-confidence 0.4 \
  --avoid-market-favorite-player-form-counter-signal \
  --market-favorite-counter-min-probability 0.6 \
  --output data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/forecast_policy_margin_0_05_market_form_counter_2026-06-04.json
```

选手状态候选：

```bash
PYTHONPATH=src python3 -m cs2pickem.cli apply-forecast-policy \
  --forecast-report data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/forecast_report.json \
  --fixtures data/cologne2026/processed/stage1_opening_fixtures_fivee_6m_model_with_market_odds_player_form_2026-06-04.csv \
  --minimum-margin 0.05 --avoid-player-form-counter-signal \
  --player-form-counter-min-confidence 0.4 \
  --avoid-player-status-risk \
  --player-status-min-confidence 0.4 \
  --player-status-min-margin 0.06 \
  --output data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/forecast_policy_margin_0_05_player_status_risk_2026-06-04.json
```

### Round 4 checkpoint

Round 4 这种中途状态先从逐场赛果推导 standings，再把 standings 合并进下一轮 fixtures，最后用 `checkpoint-pickem` 看槽位是否还活着，不用 `backtest-pickem` 提前算最终分：

```bash
PYTHONPATH=src python3 -m cs2pickem.cli standings-from-results \
  --results data/cologne2026/source_inputs/stage1_round1_4_results_2026-06-05.csv \
  --source esportsgg+hltv_major \
  --output data/cologne2026/source_inputs/stage1_round4_standings_2026-06-05.csv

PYTHONPATH=src python3 -m cs2pickem.cli merge-standings \
  --fixtures data/cologne2026/source_inputs/stage1_round5_fixtures_2026-06-05.csv \
  --standings data/cologne2026/source_inputs/stage1_round4_standings_2026-06-05.csv \
  --output data/cologne2026/processed/stage1_round5_fixtures_with_standings_2026-06-05.csv

PYTHONPATH=src python3 -m cs2pickem.cli checkpoint-pickem \
  --pickems data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/final_fused_pickem_2026-06-01.json \
  --standings data/cologne2026/source_inputs/stage1_round4_standings_2026-06-05.csv \
  --output data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/final_fused_pickem_checkpoint_round4_2026-06-05.json
```

### 完赛后 Pick'em 评分

Stage 1 全部结束后，先整理最终瑞士轮战绩 CSV，字段至少包含 `team,wins,losses`。不要用 Round 3/Round 4 的中途战绩喂给最终回测。

```csv
team,wins,losses
BetBoom,3,0
B8,3,0
...
```

然后对 README 归档的最终融合选择直接打分：

```bash
PYTHONPATH=src python3 -m cs2pickem.cli backtest-pickem \
  --pickems data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/final_fused_pickem_2026-06-01.json \
  --results data/cologne2026/source_inputs/stage1_final_standings_2026-06-XX.csv \
  --pass-threshold 5 \
  --output data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/final_fused_pickem_backtest_2026-06-XX.json
```

`backtest-pickem` 会把 `3-0`、`advance`、`0-3` 分开计分，并输出总命中数、是否达到 5 分通过线，以及每个槽位的正确/失效队伍。

## 数据输入字段

CSV 或字典行建议包含以下列；`readiness` 的字段完整性门槛会要求核心建模列已填充：

- 基础：`date` `event` `event_tier` `status` `team1` `team2` `winner` `best_of` `map`
- 队伍：`team{1,2}_rank` `team{1,2}_rmr_points` `team{1,2}_major_best_placement`
- 近期状态：`team{1,2}_matches_30d` `team{1,2}_recent_winrate_{5,10}` `team{1,2}_bo{1,3}_winrate_6m` `team{1,2}_current_streak`
- 地图：`team{1,2}_map_winrate`
- 选手：`team{1,2}_rating` `team{1,2}_kd` `team{1,2}_opening_success` `team{1,2}_clutch_winrate` `team{1,2}_star_rating` `team{1,2}_substitute_flag` `team{1,2}_player_sample` `team{1,2}_player_form_score` `team{1,2}_player_form_trend` `team{1,2}_player_sample_confidence`
- 对局：`h2h_team1_winrate` `swiss_round` `team{1,2}_wins` `team{1,2}_losses` `version_tag` `source_match_url`
- 市场信号：`odds_team1` `odds_team2`、`team1_odds` `team2_odds`、`decimal_odds_team1` `decimal_odds_team2`、`odds_team1_american` `odds_team2_american`、`market_probability_team1`、`market_signal_basis`、`market_signal_source`、`market_signal_proxy`、`hltv_poll_team1`、`hltv_poll_team2`
- BP 情报：`date` `source` `team1` `team2` `map`/`confirmed_map`/`expected_map` `confidence` `team{1,2}_bans` `team{1,2}_pick`

## 项目结构

```text
CS-AI-bet/
├── src/cs2pickem/            # 核心 Python 包（零三方依赖即可运行）
│   ├── cli.py                # 命令行入口
│   ├── pipeline.py / workflow.py   # 一键离线工作流编排
│   ├── cleaning.py           # 数据清洗
│   ├── enrichment.py         # 无泄漏滚动特征 + 队伍地图画像
│   ├── reliability.py        # 在线赛前 Elo + 不稳定身份特征屏蔽
│   ├── calibration.py        # Platt 概率校准
│   ├── tuning.py             # optimize-matches 回归验证与候选调参
│   ├── features.py / selection.py / imbalance.py
│   ├── models.py / predictor.py / evaluation.py / splitting.py
│   ├── swiss.py / pickem.py / strategy.py / forecast.py
│   ├── sources.py / update.py / fivee.py / dataset_store.py
│   ├── odds.py / players.py / bp.py / maps.py / ratings.py
│   └── readiness.py / backtest.py / export.py / visualization.py
├── tests/                    # 单元测试（标准库 unittest）
├── examples/                 # 样例 CSV/HTML/JSON 输入
├── data/                     # IEM Cologne 2026 真实数据与预测产物
│   ├── cologne2026/          # raw / processed / manifests / predictions / source_inputs
│   └── fivee/                # 5EPlay 抓取数据
├── docs/                     # 图片、技术说明和历史实施计划
├── pyproject.toml
└── README.md                 # 玩家版首页
```

## 模块说明

| 模块 | 职责 |
| --- | --- |
| `sources` / `update` | HTTP 缓存、HLTV-like 结果/event/ranking/选手页解析、版本标签、数据集 manifest |
| `daily-update` | 配置驱动的多源每日增量更新，去重追加长期训练 CSV |
| `dataset_store` | 长期训练 CSV 增量追加、去重、覆盖范围 manifest |
| `enrichment` | 从原始流水生成无未来泄漏的滚动状态特征与队伍地图画像 |
| `reliability` | 按赛前时间线注入 Elo，过滤高泄漏/高漂移身份特征 |
| `calibration` / `tuning` | Platt 校准、滚动验证、候选模型/Top-K/Elo/市场权重对比 |
| `forecast` | 赛前 fixtures 单场胜率、真实赔率修正、选手状态摘要、低置信规避、未知地图 Top3 均值预测；`apply-forecast-policy` 可在不重训时重打标策略 |
| `bp` | 合并赛前地图 BP 情报，确认地图后改用确认地图特征 |
| `odds` | 多平台赔率归一化、source URL 优先匹配、市场概率与 proxy 信号审计 |
| `players` | 按赛前 lookback 窗口把选手统计聚合成队伍级特征、短期 form、趋势和样本置信度 |
| `readiness` | 上线前数据量、字段完整性、模型指标与融合优势审计 |
| `selection` | 低方差过滤、Pearson 相关冗余过滤、按标签相关性保留 TOP-K 特征 |
| `imbalance` | 确定性 SMOTE-like 上采样与类别权重，训练/CV/预测共用 |
| `maps` | 未知 BP 时按双方 ban/pick 偏好与地图胜率生成 Top3 候选并取均值 |
| `visualization` | 从训练报告导出特征重要性图、预测概率分布图与 manifest |
| `export` | 从 Pick'em/readiness 报告生成最终答案单、选择边际与警告摘要 |
| `swiss` / `pickem` / `strategy` | 瑞士轮模拟、Pick'em 清单生成与风险感知选择策略 |
| `workflow` / `pipeline` | 一键离线编排训练、预测、模拟、审计与答案单输出 |

## 设计原则

- 离线优先：核心链路不伪造实时 HLTV 结果、赔率或选手状态；赛前需先采集干净数据再训练/模拟。
- 可复现：所有切分、采样、模拟均确定性可控，报告保留超参数、后端、边界与市场信号来源，便于审计回放。
- 优雅降级：可选依赖缺失时自动回退纯 Python 实现，脚本始终可跑。
- 市场信号克制使用：真实赔率只轻量修正，民调 proxy 只进入报告；最终专家/市场融合权重低于模型权重。

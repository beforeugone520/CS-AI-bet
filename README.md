# CS2 Major Pick'em 机器学习系统

这是一个离线可运行的 CS2 Major Pick'em 预测项目骨架，覆盖目标方案里的核心链路：

1. 数据清洗：过滤二队、临时 mix/stand-in 队、弃赛/重开、低级别赛事、90 天外低价值样本，剔除 Rating/KD 3σ 外异常值，并填充 H2H 中性默认值。
2. 特征工程：生成排名差、RMR 差、历史 Major 最佳成绩差、近期胜率差、地图胜率差、地图 one-hot、队伍/赛事标签编码、选手 Rating/KD/首杀/残局差、赔率、BO1/BO3、瑞士轮状态和晋级/淘汰距离等特征。
3. 时间切分：按时间顺序拆分训练/验证/测试集，并提供时间序列交叉验证折。
4. 模型融合：提供逻辑回归、深度受限随机森林风格树集成、Boosting 风格残差树、轻量神经网络风格模型，默认权重为 20%/30%/35%/15%，并按目标配置记录学习率、树数、最大深度、最小叶子样本、子采样和隐藏层。
5. 瑞士轮模拟：按种子蛇形配对、同战绩优先、尽量避免复赛，晋级/淘汰战自动按 BO3 处理。
6. Pick'em 策略：输出 3-0、晋级、0-3 候选，并应用赔率修正、低置信规避、弱队爆冷降权和挑战者/传奇组分层策略。

当前实现不依赖第三方包，可以直接用系统 Python 验证。真实爬虫、pandas/sklearn/XGBoost/TensorFlow 训练可通过 `pyproject.toml` 的可选依赖继续扩展。
安装 `ml` 可选依赖后，默认融合模型会优先使用 scikit-learn 的 LogisticRegression/RandomForest/MLP、XGBoost 的 XGBClassifier，并用 joblib 并行训练各基础模型；依赖缺失或样本不适配时会自动回退到纯 Python 模型，保持离线脚本可运行。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m cs2pickem.cli demo
PYTHONPATH=src python3 -m cs2pickem.cli enrich --matches examples/raw_match_history.csv --output /tmp/cs2pickem_enriched.csv --profiles-output /tmp/cs2pickem_profiles.json
PYTHONPATH=src python3 -m cs2pickem.cli train --matches examples/sample_matches.csv --reference-date 2026-05-31 --top-k 25 --cv-folds 5 --max-age-days 180 --output /tmp/cs2pickem_train_report.json
PYTHONPATH=src python3 -m cs2pickem.cli visualize --training-report /tmp/cs2pickem_train_report.json --output-dir /tmp/cs2pickem_viz --prefix sample
PYTHONPATH=src python3 -m cs2pickem.cli readiness --matches examples/sample_matches.csv --training-report /tmp/cs2pickem_train_report.json --participants examples/major_participants_sample.csv --top-teams examples/top80_teams_sample.csv --minimum-max-age-days 180 --require-validation-tuned-weights --forecast-report /tmp/cs2pickem_pipeline/forecast_report.json --require-forecast-low-confidence-avoidance --pickem-report /tmp/cs2pickem_pipeline/pickem_report.json --minimum-pickem-simulations 100000 --require-pickem-slots --minimum-pickem-selection-margin 0.04 --minimum-pickem-market-adjusted-matchups 1 --source-manifest /tmp/cs2pickem_daily_update/daily_update_manifest.json --required-source hltv-results --source-reference-time 2026-06-01T00:00:00+00:00 --maximum-source-age-hours 24
PYTHONPATH=src python3 -m cs2pickem.cli merge-odds --matches examples/upcoming_fixtures.csv --odds examples/odds_feed.csv --output /tmp/cs2pickem_fixtures_with_odds.csv
PYTHONPATH=src python3 -m cs2pickem.cli merge-players --matches /tmp/cs2pickem_fixtures_with_odds.csv --players examples/player_stats.csv --output /tmp/cs2pickem_fixtures_ready.csv --window-days 15
PYTHONPATH=src python3 -m cs2pickem.cli merge-bp --fixtures /tmp/cs2pickem_fixtures_ready.csv --bp examples/bp_intel.csv --output /tmp/cs2pickem_fixtures_with_bp.csv
PYTHONPATH=src python3 -m cs2pickem.cli forecast --history /tmp/cs2pickem_enriched.csv --fixtures /tmp/cs2pickem_fixtures_with_bp.csv --profiles /tmp/cs2pickem_profiles.json --reference-date 2026-05-31 --top-k 25 --max-age-days 180
PYTHONPATH=src python3 -m cs2pickem.cli pickem --history /tmp/cs2pickem_enriched.csv --teams examples/sample_teams.csv --fixtures /tmp/cs2pickem_fixtures_with_bp.csv --profiles /tmp/cs2pickem_profiles.json --reference-date 2026-05-31 --simulations 100000 --stage challengers --max-age-days 180
PYTHONPATH=src python3 -m cs2pickem.cli answer-sheet --pickem-report /tmp/cs2pickem_pipeline/pickem_report.json --readiness-report /tmp/cs2pickem_pipeline/readiness_report.json --output /tmp/cs2pickem_pipeline/pickem_answer_sheet.json
PYTHONPATH=src python3 -m cs2pickem.cli backtest-pickem --pickems /tmp/cs2pickem_pipeline/pickem_report.json --results examples/final_standings_sample.csv --pass-threshold 5 --output /tmp/cs2pickem_pickem_backtest.json
PYTHONPATH=src python3 -m cs2pickem.cli backtest-pickem-suite --suite examples/pickem_backtest_suite_sample.json --pass-rate-target 0.38 --output /tmp/cs2pickem_pickem_backtest_suite.json
PYTHONPATH=src python3 -m cs2pickem.cli simulate --teams examples/sample_teams.csv --simulations 1000
PYTHONPATH=src python3 -m cs2pickem.cli update --html examples/hltv_results_fixture.html --version-log examples/version_log.csv --output /tmp/cs2pickem_matches.json --manifest /tmp/cs2pickem_manifest.json --dataset /tmp/cs2pickem_training_matches.csv --dataset-manifest /tmp/cs2pickem_training_manifest.json
PYTHONPATH=src python3 -m cs2pickem.cli daily-update --config examples/daily_update_config.json --output-dir /tmp/cs2pickem_daily_update
PYTHONPATH=src python3 -m cs2pickem.cli event-teams --html examples/hltv_event_fixture.html --output /tmp/cs2pickem_participants.csv --manifest /tmp/cs2pickem_participants_manifest.json
PYTHONPATH=src python3 -m cs2pickem.cli rankings --html examples/hltv_rankings_fixture.html --output /tmp/cs2pickem_top80.csv --manifest /tmp/cs2pickem_top80_manifest.json --limit 80
PYTHONPATH=src python3 -m cs2pickem.cli player-stats --html examples/hltv_player_stats_fixture.html --date 2026-05-31 --output /tmp/cs2pickem_player_stats.csv --manifest /tmp/cs2pickem_player_stats_manifest.json
PYTHONPATH=src python3 -m cs2pickem.cli fivee-collect --url case --output-dir /tmp/cs2pickem_fivee --cache-dir /tmp/cs2pickem_fivee_cache --delay-seconds 0
PYTHONPATH=src python3 -m cs2pickem.cli fivee-match-results --start-date 2025-12-01 --end-date 2026-06-01 --output-dir /tmp/cs2pickem_fivee_results --cache-dir /tmp/cs2pickem_fivee_results_cache --page-size 100 --max-pages 500
PYTHONPATH=src python3 -m cs2pickem.cli pipeline --history examples/raw_match_history.csv --fixtures examples/upcoming_fixtures.csv --teams examples/sample_teams.csv --odds examples/odds_feed.csv --players examples/player_stats.csv --bp examples/bp_intel.csv --participants examples/major_participants_sample.csv --top-teams examples/top80_teams_sample.csv --version-log examples/version_log.csv --pickem-backtest-report /tmp/cs2pickem_pickem_backtest_suite.json --pickem-pass-rate-target 0.38 --source-manifest /tmp/cs2pickem_daily_update/daily_update_manifest.json --required-source hltv-results --source-reference-time 2026-06-01T00:00:00+00:00 --maximum-source-age-hours 24 --reference-date 2026-05-31 --output-dir /tmp/cs2pickem_pipeline --simulations 100000 --stage challengers --max-age-days 180
PYTHONPATH=src python3 -m cs2pickem.cli pipeline --history examples/raw_match_history.csv --fixtures examples/cologne2026_stage1_opening_fixtures.csv --teams examples/cologne2026_stage1_teams.csv --participants examples/cologne2026_stage1_teams.csv --top-teams examples/cologne2026_stage1_teams.csv --version-log examples/version_log.csv --reference-date 2026-06-01 --output-dir /tmp/cs2pickem_cologne2026_stage1 --simulations 100000 --stage challengers --max-age-days 180
PYTHONPATH=src python3 -m cs2pickem.cli readiness --matches /tmp/cs2pickem_pipeline/enriched_matches.csv --training-report /tmp/cs2pickem_pipeline/train_report.json --participants examples/cologne2026_participants.csv --top-teams examples/cologne2026_participants.csv --minimum-max-age-days 180 --sample-reference-date 2026-06-01 --maximum-sample-age-days 180
```

`enrich` 会把原始比赛流水按日期滚动生成训练字段：近 5/10 场胜率、30 天参赛量、BO1/BO3 胜率、地图胜率、近 3 次 H2H、当前连胜/连败，并输出队伍地图偏好/禁图画像。
`train` 会输出清洗后样本量、时间序列切分计数、防泄漏日期边界、特征名、筛选特征、特征重要性、样本不平衡处理报告、默认融合权重、基础模型超参数、基于验证集 Log Loss 的赛前微调权重、Accuracy/AUC/Log Loss/盈亏、5 折时间序列 CV、BO1/BO3 分段指标、单模型与融合模型对比，以及测试集每场 `team1` 胜率；样本太少时会降级为 in-sample 报告并保留同一 JSON 结构。默认清洗窗口为 90 天，赛前按 2026.01-2026.05 六个月拆分时传 `--max-age-days 180`；要严格复现目标里的 2026.01-2026.04 / 2026.05 上旬 / 2026.05 下旬边界，在真实六个月训练集上加 `--train-end-date 2026-04-30 --validation-end-date 2026-05-15`。显式日历拆分会要求 train/validation/test 三段都有数据，避免样例数据被错误边界切成伪评估。
`visualize` 会从训练报告生成特征重要性和预测概率分布图，优先使用 Matplotlib/Agg 输出 PNG；未安装 Matplotlib 时自动输出 SVG。
`readiness` 会审计训练数据和训练报告是否满足上线门槛：≥8000 条、队伍覆盖、参赛名单/Top 队伍名单覆盖、S/A 级近窗口高质量样本范围、完整建模字段、BO1/BO3 性能目标、融合模型是否优于单模型、训练窗口是否满足要求、验证集 Log Loss 调权是否完成、历史 Pick'em 回测通过率、单场低置信预测是否规避、Pick'em Swiss 蒙特卡洛模拟次数、最终 `3-0/advance/0-3` 答案槽位完整性、Pick'em 选择边际、市场赔率是否实际进入 Pick'em 模拟，以及赛前数据源 manifest 是否在 24 小时内刷新；真实六个月生产训练可加 `--expected-train-end-date 2026-04-30 --expected-validation-end-date 2026-05-15 --minimum-max-age-days 180 --sample-reference-date 2026-05-31 --maximum-sample-age-days 180 --require-validation-tuned-weights --forecast-report /tmp/cs2pickem_pipeline/forecast_report.json --require-forecast-low-confidence-avoidance --pickem-report /tmp/cs2pickem_pipeline/pickem_report.json --minimum-pickem-simulations 100000 --require-pickem-slots --minimum-pickem-selection-margin 0.04 --minimum-pickem-market-adjusted-matchups 1 --source-manifest /tmp/cs2pickem_daily_update/daily_update_manifest.json --required-source hltv-results --source-reference-time 2026-06-01T00:00:00+00:00 --maximum-source-age-hours 24`，强制检查目标日历拆分、行级采集范围、赛前验证集调权、50%-52% 低置信单场全部输出 `avoid`、10 万次模拟门槛、完整 Pick'em 清单、每个入选项相对下一候选至少 4% 的边际、至少一个 Swiss matchup 使用真实赔率修正，以及关键赛前源不是过期快照。
`merge-odds` 会把多平台十进制赔率按日期和无序队伍对齐，自动处理队伍顺序反转，输出均值赔率、provider 数量和归一化市场概率。
`merge-players` 会按比赛日期只读取赛前窗口内的选手行，合并 Rating/KD/首杀成功率/残局胜率、明星选手 Rating、替补标记和样本量，避免未来数据泄漏。
`merge-bp` 会按日期和无序队伍合并赛前地图 BP 情报，覆盖确认地图并保留来源、置信度、双方禁图/选图字段；队伍顺序反转时会自动交换 team-scoped 字段。
`forecast` 会用历史训练表预测赛前 fixtures，输出四个基础模型的独立胜率、融合权重、基础模型超参数、加权贡献、模型原始胜率、赔率修正后胜率、市场修正是否应用、候选地图分布、`confidence_margin`、低置信标记、`Alpha/Bravo/avoid` 级别的单场决策和 `decision_summary`；只有 fixture 同时提供真实 `odds_team1/odds_team2` 时才做 ±3% 赔率修正，缺失赔率时保留模型原始概率；也可直接传 `--bp examples/bp_intel.csv` 在预测前应用 BP 情报。
`pickem` 会用同一套融合模型作为 Swiss 每一场的胜率函数，并在进入蒙特卡洛前按 `--fixtures` 中的对局级真实赔率做 ±3% 市场情绪修正，队伍顺序反转时会自动换边；若没有 fixtures 赔率，会兼容读取 team CSV 中的旧式 `odds` 字段，缺失赔率时不会用默认赔率伪造市场信号，输出 3-0、晋级、0-3 Pick'em 清单；报告会保留每个入选 Pick'em 的类别概率、排名、策略选择分数、下一候选分数和选择边际，也会保留 `pickem_risk_details`，逐队解释基础概率、阶段加成、相对最高排名的落差、黑马惩罚倍数和最终选择分数；因此 readiness 的 Pick'em selection margin 门槛会按真实策略分数审计，而不是只看未加权原始概率；同时保留基础模型超参数、若干样例对局的模型原始概率、赔率修正后概率、市场修正是否应用、市场修正命中 matchup 摘要和基础模型分解，便于追踪模拟输入；`--stage challengers` 会提高 BO1/地图池权重，`--stage legends` 会提高排名、Rating 稳定性和深轮次强度权重，排名落差超过 15 名的 3-0/晋级候选会被降低选择权重。
`answer-sheet` 会把大型 Pick'em 报告和 readiness 报告压缩成可提交/复核的最终答案单：保留 `3-0/advance/0-3` 槽位、按提交顺序展开每个 pick 的概率/排名/策略分数/选择边际，并输出按晋级概率排序的 `team_outcomes` 表，包含每队 `3-0/3-1/3-2/0-3/1-3/2-3/advance/eliminate` 概率和最可能瑞士轮战绩；同时汇总 readiness 失败项、最小选择边际、赔率修正覆盖数和警告；`pipeline` 会自动写出同样的 `pickem_answer_sheet.json`。
`backtest-pickem` 会把 Pick'em 报告和最终 Swiss standings CSV 对齐，计算 3-0/晋级/0-3 命中数、是否达到 pass threshold；模块内也提供 suite 级 pass rate 汇总，用于验证历史同类 Major 回测通过率是否达到 38% 目标。
`simulate` 会输出每队 `3-0/3-1/3-2/0-3/1-3/2-3/advance/eliminate` 概率和 Pick'em 候选。
`update` 可从本地 HTML 或 `--url` 抓取并缓存 HLTV-like 结果页，写出 JSON 数据集和 manifest；真实使用时建议把 `--cache-dir` 指到持久目录，定时运行并保留 manifest。默认 URL 抓取会按 urllib、可选 `requests`、curl CLI 的顺序回退；HLTV 如果返回 Cloudflare challenge，先用浏览器保存 HTML，再走 `--html` 导入。
`daily-update` 可读取 JSON 配置批量运行多个结果来源任务，统一写 per-job 产物、总 manifest，并把新增比赛去重追加到长期训练 CSV；适合接入 cron/launchd 做每日增量更新。
`event-teams` 可从本地 HTML 或 `--url` 抓取 HLTV-like event 页面，解析参赛队伍、种子、世界排名和晋级来源，输出可直接传给 `readiness --participants` 的 CSV。
`rankings` 可从本地 HTML 或 `--url` 抓取 HLTV-like ranking 页面，解析 Top-N 队伍、排名、积分和区域，输出可直接传给 `readiness --top-teams` 的 CSV；默认 `--limit 80` 对齐目标里的全球 Top80 覆盖。
`player-stats` 可从本地 HTML 或 `--url` 抓取 HLTV-like 选手统计页面，解析 Rating、KD、首杀成功率、残局胜率和替补标记，输出可直接传给 `merge-players` 的 CSV。
`fivee-collect` 会低频抓取 5E 公开战队页（可传完整 URL、`/data/team/<slug>` 或裸 slug），缓存原始 HTML，并输出 `fivee_teams.csv`、`fivee_players.csv`、`fivee_maps.csv` 和 `fivee_manifest.json`；当前解析公开页可见的战队总览、队员列表和地图分析，遇到 WAF/验证码页会在 manifest 里标为 blocked，便于改用浏览器保存 HTML 后再接入。
`fivee-match-results` 会从 5E 公开赛事结果接口反向翻页抓取指定日期窗口内的全局 CS2 赛果，缓存 JSON，并输出 `fivee_match_results.csv`、`fivee_match_maps.csv` 和 `fivee_match_results_manifest.json`；适合直接抓 6 个月训练/预测窗口，再按参赛队伍过滤成模型特征。
`update`/`daily-update` 可在写入长期训练 CSV 前用 `team_metadata`/`--team-metadata` 补排名、RMR、Major 历史字段，用 `player_stats`/`--players` 按赛前窗口补选手字段，并可用 `default_swiss_state`/`--default-swiss-state` 给普通结果填充中性 Swiss 状态；`daily-update` 配置提供 `participants`、`top_teams`、`minimum_rows`、`required_teams` 时，会在总 manifest 里输出长期数据集距离 ≥8000 行、Top80 和 Major 参赛覆盖的缺口。
`pipeline` 会把 enrich、训练历史字段增强、fixtures 字段增强、merge-odds、merge-players、merge-bp、train、visualize、forecast、pickem、readiness 串成一次离线赛前工作流；训练历史会用 `--teams` 补队伍静态字段、用 `--players` 补赛前选手窗口字段、用 `--version-log` 补版本标签，赛前 fixtures 也会在预测前用 `--teams` 补排名等静态字段并填充瑞士轮初始状态；增强后的 fixtures 会同时传给 forecast 和 pickem，因此 `--odds` 合并出的对局级赔率会进入 Swiss 蒙特卡洛胜率函数；传入 `--pickem-backtest-report` 时，readiness 会把历史 Major Pick'em suite 通过率纳入生产门槛，默认目标为 38%；pipeline readiness 默认还要求验证集 Log Loss 调权已完成、forecast 中低置信单场全部为 `avoid`、Pick'em 报告中的 Swiss 蒙特卡洛模拟次数 ≥100000、最终 `3-0/advance/0-3` 槽位达到 `2/6/2`、没有重复队伍，且每个 Pick'em 入选项的选择边际 ≥0.04；当传入 `--odds` 时，pipeline 还会默认要求至少 1 个 Pick'em Swiss matchup 使用真实赔率修正，可用 `--minimum-pickem-market-adjusted-matchups` 调整；传入 `--source-manifest`、`--required-source` 和 `--maximum-source-age-hours` 时，pipeline readiness 也会拦截超过赛前窗口的旧数据源快照；也可用 `--minimum-pickem-simulations` 和 `--minimum-pickem-selection-margin` 调整其他门槛，或用 `--skip-pickem-slot-check` 跳过槽位检查；训练报告中的验证集微调权重会传给 forecast/pickem，训练、forecast、pickem 三份报告都会写出同一套基础模型超参数，最后写出 `pipeline_manifest.json`。
加上 `--dataset` 后，`update` 会把解析结果增量追加到长期训练 CSV，并按 `source_match_url` 或日期+队伍+地图去重，同时写出覆盖日期、队伍数和来源的 dataset manifest。

## 数据输入字段

CSV 或字典行建议包含；`readiness` 的字段完整性门槛会要求下列核心建模列已填充：

- 基础：`date`, `event`, `event_tier`, `status`, `team1`, `team2`, `winner`, `best_of`, `map`
- 队伍：`team1_rank`, `team2_rank`, `team1_rmr_points`, `team2_rmr_points`, `team1_major_best_placement`, `team2_major_best_placement`
- 近期状态：`team1_matches_30d`, `team2_matches_30d`, `team1_recent_winrate_5`, `team2_recent_winrate_5`, `team1_recent_winrate_10`, `team2_recent_winrate_10`, `team1_bo1_winrate_6m`, `team2_bo1_winrate_6m`, `team1_bo3_winrate_6m`, `team2_bo3_winrate_6m`, `team1_current_streak`, `team2_current_streak`
- 地图：`team1_map_winrate`, `team2_map_winrate`
- 选手：`team1_rating`, `team2_rating`, `team1_kd`, `team2_kd`, `team1_opening_success`, `team2_opening_success`, `team1_clutch_winrate`, `team2_clutch_winrate`, `team1_star_rating`, `team2_star_rating`, `team1_substitute_flag`, `team2_substitute_flag`, `team1_player_sample`, `team2_player_sample`
- 对局：`h2h_team1_winrate`, `odds_team1`, `odds_team2`, `swiss_round`, `team1_wins`, `team1_losses`, `team2_wins`, `team2_losses`, `version_tag`
- BP 情报：`date`, `source`, `team1`, `team2`, `map`/`confirmed_map`/`expected_map`, `confidence`, `team1_bans`, `team2_bans`, `team1_pick`, `team2_pick`

## 已实现模块

- `sources/update`：HTTP 缓存、HLTV-like 结果页解析、event 参赛队伍解析、Top-N ranking 解析、选手统计解析、版本日志标签、数据集 manifest。
- `daily-update`：配置驱动的多来源每日增量更新，统一追加长期训练 CSV 并输出 per-job/总 manifest。
- `dataset_store`：长期训练 CSV 增量追加、去重、覆盖范围 manifest。
- `enrichment`：从原始比赛流水生成无未来泄漏的滚动状态特征和队伍地图画像。
- `forecast`：对赛前 fixtures 输出单场胜率、赔率修正、低置信规避和未知地图 Top3 均值预测。
- `bp`：合并赛前地图 BP 情报，确认地图后跳过未知地图 Top3 均值预测，改用确认地图特征。
- `odds`：多平台赔率归一化、反向队伍匹配、均值合并和覆盖率报告。
- `players`：按赛前 lookback 窗口把选手级统计聚合成队伍级训练/预测特征。
- `readiness`：上线前数据量、字段完整性、模型指标和融合模型优势审计。
- `selection`：低方差过滤、Pearson 相关冗余过滤、按标签相关性保留 TOP-K 特征。
- `imbalance`：确定性 SMOTE-like 少数类上采样和类别权重，训练、CV、预测器共用。
- `maps`：未知 BP 时按双方 ban/pick 偏好和地图胜率生成 Top3 候选地图，并取多地图预测均值。
- `visualization`：从训练报告导出特征重要性图、预测概率分布图和 manifest。
- `export`：从 Pick'em/readiness 大报告生成最终答案单、选择边际和警告摘要。
- `workflow`：一键离线编排训练、赛前预测、Swiss 模拟、readiness 和最终答案单输出。

## 说明

这个版本完成的是可执行核心，不伪造实时 HLTV 结果、赔率或选手状态。赛前使用时，应先把 HLTV/5E/VPGAME/赔率源采集后的干净数据导入，再运行训练和模拟。
`examples/cologne2026_participants.csv` 是 2026-06-01 核对的 32 队全量参赛快照，包含 Stage 1/2/3 起始阶段和 VRS/RMR 分数；`examples/cologne2026_stage1_teams.csv` 与 `examples/cologne2026_stage1_opening_fixtures.csv` 是 Stage 1 专用输入快照，用于验证真实队名/首轮 fixtures 的管线兼容性。它们不是长期训练数据，也不替代赛前最新抓取、赔率和选手状态更新。

# WF-2 建模升级与集成收尾报告

> 时间窗口：2026-06（提交链 `WF-2A` → `WF-2F`，集成收尾 `WF-2G`）。
> 适用范围：`src/cs2pickem/` Python ML 工具链。前端 `site/` 与本次升级无耦合，未被触碰。
> 诚实口径：本报告**只把有证据支持的两项**翻成生产默认；未证明 / 回归 / 不可证伪的轴一律如实记录为「实现但未默认启用，待多赛季或回合分语料」，不粉饰为全面提升。

---

## 1. 动机与背景

### 1.1 现状测绘

升级前的 `cs2pickem` 是一套纯 Python、零核心依赖的离线建模工具链（可选 numpy/scikit-learn/xgboost/joblib 加速，缺失即回退纯 Python）。其建模层只有三块：

- **评级**：在线赛前 **Elo**（按比赛日期滚动，无泄漏）。
- **概率校准**：验证集 **Platt** 校准。
- **市场融合**：`legacy_clip` —— 把模型概率向市场概率做 ±`max_adjustment` 的算术微调。

这套链路在赛果锁定、瑞士轮模拟、Pick'em 风险分层上已经成熟，但建模质量层相对单薄，与公开 SOTA 有明显缺口。

### 1.2 SOTA 缺口

1. **评级**：Elo 不建模评级不确定性；Glicko-2 用 RD（rating deviation）显式表达「这个队最近打得少 / 刚换人，评级不可信」，对样本稀疏的 Major 语境更合适。
2. **概率融合**：算术 clip 不是概率论上正确的专家合并；**对数意见池**（logit pooling / 几何平均共识）才是把两个概率专家合并的标准做法。
3. **未用信号**：已抓但未用的 Bradley-Terry 实力先验、de-vig 公平概率（去掉博彩抽水的 overround）。
4. **赛制**：瑞士轮配对只有蛇形种子，缺 Valve Major 的 Buchholz 难度再排序。
5. **下注**：完全没有 bankroll 管理（Kelly 分注）。

### 1.3 红线约束

整个 WF-2 在以下硬约束下进行，任何一条破坏即视为不可接受：

- **零新第三方依赖**，纯 Python 可端到端跑，可选加速保留纯 Python 回退。
- **无未来泄漏**：所有新评级 / 特征 / de-vig 按比赛日期严格截断。
- **train/serve 一致**：拟合期注入的列，serve 期必须用同一引擎注入，否则 train/serve skew 比不注入还差。
- **只翻有证据支持的默认**：用同口径回测 + 显著性裁决，no_significant_diff / 回归 / 不可证伪的轴一律默认关闭、仅 opt-in。

---

## 2. 分阶段实施（WF-2A ~ WF-2F 提交链）

| 提交 | 主题 | 做了什么 |
| --- | --- | --- |
| `WF-2A` | 地基：修泄漏 + 评估严谨层 | 修回测中已存在的泄漏点；新增 paired bootstrap、Diebold-Mariano 检验、分箱 ECE 等评估严谨原语（`evaluation.py`），为后续所有 A/B 提供统计判据 |
| `WF-2B` | 信号接入 | Bradley-Terry 实力先验（`reliability.apply_final_bt_to_match`，可与任一评级并存）、de-vig（`odds.devig_market`：multiplicative / power / shin 三法 + overround/devig_z/devig_power_k 审计字段）、把「已抓但未用」特征接进特征准备 |
| `WF-2C` | 评级核心 | Glicko-2（`ratings.py`）：赛前按周期批量更新、MOV（margin-of-victory）阻尼、不活跃 RD 膨胀；写出 `glicko_diff`（反对称赛前评级差）/ `glicko_rd_sum`（对称不确定性）候选列，全程无泄漏 |
| `WF-2D` | 概率质量 | 多方法校准（`calibration.py`：platt / beta / temperature）、logit 市场融合（`strategy._logit_pool`）、分数 Kelly 下注 + 组合敞口上限（`staking.py`，仅半 Kelly，劝阻 >0.5） |
| `WF-2E` | 赛制 + EV | Buchholz 配对（`swiss.simulate_swiss(pairing='buchholz')`，1v9 开局 + (wins,losses) 桶按 Buchholz 难度再排序）、series BO3 提升层（`series.py`：逐图→系列闭式合成）、Pick'em EV 多目标（`pickem`：`expected_hits` / `threshold_prob` / `leveraged`） |
| `WF-2F` | 大回测裁决 | 同口径 A/B（统一 split / 特征 / 校准口径）+ 显著性（paired bootstrap + DM），逐轴产出采纳 / 未证明 / 回归 / 不可证伪判定 |

---

## 3. WF-2F 裁决矩阵（真实回测数字）

回测语料：`train_match_level_holdout`（n=1301，模型层）与 `odds_subset_pooled_walkforward`（n=515，赔率融合层）。split：train 10412 / val 1301 / test 1301。
约定：`mean Δlogloss = baseline_loss − config_loss`，**正值表示候选更好**。显著性以 paired bootstrap 95% CI 不跨 0 且 DM p 小为准。

### 3.1 模型层（baseline logloss = 0.680627，n=1301）

| 配置 | candidate logloss | mean Δlogloss | 95% CI | DM p | 裁决 |
| --- | --- | --- | --- | --- | --- |
| **plus_glicko** | 0.669998 | **+0.010629** | [+0.0042, +0.0168] | 0.000715 | **显著改进 → 采纳为默认评级** |
| plus_bt | 0.681438 | −0.000810 | [−0.0022, +0.0005] | 0.237542 | no_significant_diff → 默认关闭 |
| cal_platt_beta | 0.681113 | −0.000486 | [−0.0017, +0.0008] | 0.446306 | no_significant_diff → 仍 platt |
| cal_platt_temperature | 0.681261 | −0.000634 | [−0.0019, +0.0007] | 0.328430 | no_significant_diff → 仍 platt |
| cal_platt_beta_temperature | 0.681467 | −0.000839 | [−0.0022, +0.0005] | 0.224337 | no_significant_diff → 仍 platt |
| plus_unverified | 0.681657 | −0.001030 | [−0.0023, +0.0002] | 0.103785 | no_significant_diff → 默认关闭 |

留一验证（baseline = `all_on_model` 全开，logloss 0.669987）：从全开里**抽掉 Glicko** 使 logloss 退到 0.681543（Δ −0.011556，p=0.000677，**显著回归**），证明 Glicko 是全开组合里唯一不可或缺的贡献者；抽掉 BT（Δ −6.9e-05，p≈0.96）/ 抽掉 unverified（Δ −0.0008，p≈0.70）均无显著影响。

### 3.2 赔率融合层（n=515）

纯模型 logloss 0.854851（ECE 0.202），纯市场 multiplicative de-vig logloss 0.554079（ECE 0.045）——市场是强的近校准聚合器。向市场融合：

| 配置 | logloss | vs 纯市场(mw=0) Δ | DM p | 说明 |
| --- | --- | --- | --- | --- |
| mw_legacy_0.15 | 0.701274 | +0.153577 | 0 | legacy_clip 算术融合 |
| mw_legacy_0.3 | 0.637061 | +0.217790 | 0 | legacy_clip @0.30 |
| mw_legacy_0.45 | 0.598940 | +0.255911 | 0 | legacy_clip @0.45 |
| **logit_pool_mw0.3** | **0.587435** | **+0.267416** | 0 | **对数意见池 @0.30（最优）** |
| logit_pool_mw0.35 | 0.599654 | +0.255197 | 0 | 对数意见池 @0.35 |
| logit_pool_mw0.5 | 0.644697 | +0.210154 | 0 | 对数意见池 @0.50 |
| logit_pool_mw0.65 | 0.699841 | +0.155010 | 0 | 对数意见池 @0.65 |

**logit_pool @0.30 对 legacy_clip @0.30 再 +0.049626（DM p=0，显著）**，且在扫过的 model_weight 网格里 0.30 最低 logloss。`best_logit_model_weight=0.3`、`best_calibration=['platt']`。

de-vig 变体（baseline = `devig_multiplicative_legacy_mw03`，logloss 0.637061）：

| 配置 | logloss | mean Δlogloss | DM p | 裁决 |
| --- | --- | --- | --- | --- |
| devig_power_legacy_mw03 | 0.645342 | −0.008281 | 0.000925 | **显著回归 → 仍 multiplicative** |
| devig_shin_legacy_mw03 | 0.641777 | −0.004716 | 0.000332 | **显著回归 → 仍 multiplicative** |

### 3.3 逐轴结论汇总

| 轴 | 裁决 |
| --- | --- |
| Glicko-2 评级 | **显著改进 → 采纳为生产默认** `rating_mode='glicko'` |
| logit_pool 融合（model_weight≈0.30） | **显著改进 → 采纳为生产融合默认** |
| Bradley-Terry `inject_bt` | no_significant_diff → 实现保留，默认关闭（opt-in） |
| 多方法校准 beta / temperature | no_significant_diff → 默认仍 platt |
| `include_unverified` 特征 | no_significant_diff → 默认关闭 |
| de-vig power / shin | **显著回归 → 默认仍 multiplicative** |
| Glicko-MOV | 不可证伪（缺回合分语料覆盖）→ 实现保留，默认沿用但记为待验证 |

---

## 4. 采纳的生产默认与接通点（诚实口径）

### 4.1 `rating_mode='glicko'`（评级默认翻转）

- 改在 `predictor.py` 的 `MatchPredictor.train`：默认参数 `rating_mode='glicko'`，`use_glicko = inject_glicko or rating_mode=='glicko'`。
- `forecast.forecast_fixtures` 与 `pickem` 两条服务路径调用 `MatchPredictor.train` 时**不显式传 `rating_mode`**，因此自动继承 glicko 默认。
- **Elo 仍并存注入**（`inject_elo=True` 不变），`glicko_diff` / `glicko_rd_sum` 与 elo 列同台进 `FeatureSelector` 竞选。
- **train/serve 配对**：拟合期把整段历史最终 Glicko 拟合快照进 `team_glicko_state`；serve 端 `predict_probability_details` 检测到非空 `team_glicko_state` 即调用 `apply_final_glicko_to_match` 给上场 fixture 注入同口径列。`team_glicko_state` 为空（`rating_mode='elo'` opt-in 路径）时该注入是 no-op，与历史字节一致。
- `rating_mode='elo'` 作为显式 opt-in 基线保留。

### 4.2 `logit_pool` @0.30（融合默认翻转）

- 在 `forecast.py` 与 `pickem.py` 两个生产融合调用点**显式传参** `fusion_method=PRODUCTION_FUSION_METHOD`、`model_weight=PRODUCTION_MODEL_WEIGHT`（`strategy.py` 集中定义 `PRODUCTION_FUSION_METHOD='logit_pool'`、`PRODUCTION_MODEL_WEIGHT=0.30`）。
- **底层库常量刻意不动**：`strategy.DEFAULT_FUSION_METHOD='legacy_clip'`、`DEFAULT_MODEL_WEIGHT=0.35` 被行为契约测试锁定，保持字节一致；裸调用 `adjust_probability_*` 仍是历史行为。生产翻转只发生在调用点。

### 4.3 0.30 vs 0.35 口径分歧的统一说明

`DEFAULT_MODEL_WEIGHT=0.35` 是写库时基于文献先验的诊断默认（「市场是强聚合器，模型权重约 1/3」），刻意**不在单赛事小 holdout 上拟合**（避免拟合噪声）。WF-2F 在 515 场赔率子集上做了 model_weight 网格扫描，实测最优落在 **0.30**（比 0.35 更偏市场），于是生产路径用 `PRODUCTION_MODEL_WEIGHT=0.30`，而库 / tuning 诊断默认仍保留 0.35。两者并存是有意为之：前者是「这批数据上的最优生产值」，后者是「不依赖单赛事拟合的稳健诊断先验」。多赛季再裁前不统一。

---

## 5. 未证明 / 回归 / 不可证伪项：实现但未默认启用

| 轴 | 现默认 | 原因 | 重新裁决条件 |
| --- | --- | --- | --- |
| Bradley-Terry `inject_bt` | 关闭 | no_significant_diff（Δ −0.0008，p≈0.24；留一抽掉也无影响） | 多赛季历史样本下重做 A/B |
| 多方法校准 beta / temperature | 仍 platt | no_significant_diff（三变体 p 均 >0.2） | 更大样本下校准曲线差异才可能显著 |
| `include_unverified` 特征 | 关闭 | no_significant_diff（Δ −0.0010，p≈0.10） | 无泄漏的历史选手 / 状态语料补全后重评 |
| de-vig power / shin | 仍 multiplicative | **回测显著回归**（power/shin 都更差） | 不同博彩 overround 结构 / 更大赔率语料下重测 |
| Glicko-MOV | MOV 阻尼随 glicko 启用但记为待验证 | **不可证伪**：现有语料缺逐回合 / 比分覆盖，无法把 MOV 项单独证伪或证实 | 回合分（round-score）语料补全后单独裁 MOV 贡献 |
| Buchholz 配对 / series BO3 / pickem `threshold_prob`·`leveraged` / Kelly | 各自沿用历史口径（legacy / 关闭 / expected_hits / 不下注） | 赛制 / EV / 下注层能力，按场景 opt-in，不属于「概率质量默认」 | 实盘 Kelly 上线需先定门槛 |

---

## 6. 回归与验证

- **全量测试在翻转后保持全绿**：`PYTHONPATH=src python3 -m unittest discover -s tests` 当前 **512 项全绿**（约束文档写的 509 是更早快照，本次实测 512，含本阶段新增的 serve 一致性与负控制测试）。
- **无数值基线测试需更新**：默认翻转走调用点显式传参 + `train` 默认值，库 / tuning 全局常量保持字节一致，因此被「数值基线」锁定的测试无需改动。
- **新增 serve 一致性守护**：`tests/test_forecast.py::test_default_rating_mode_glicko_pairs_train_and_serve_injection` 锁 train/serve 配对——断言默认路径下 `feature_preparation['glicko'].basis=='chronological_pre_match_rolling'`、`team_glicko_state` 非空、且 serve 端 `apply_final_glicko_to_match` 注入的 `glicko_diff != 0.0`（且符号正确）。`test_rating_mode_elo_opt_in_leaves_glicko_unapplied` 是负控制：`rating_mode='elo'` 时两侧都不注入 Glicko。
- **数值基线测试 vs 行为契约测试的区分原则**：数值基线测试钉死「某输入下的具体输出数字」，只在算法数值口径真变时才更新；行为契约测试钉死「库默认 / 公共接口 / 不变量」（如 `DEFAULT_FUSION_METHOD`、view 渲染契约），生产翻转不得破坏它们——所以默认通过调用点显式传参实现，而非改全局常量。

---

## 7. 文档同步与交接

| 文件 | 记录项 |
| --- | --- |
| `AGENTS.md` | 新增「WF-2 评级 / 融合默认与 opt-in 清单」节：两个生产默认、train/serve 配对铁律、opt-in 表格；验证命令旁注 512 项全绿 |
| `CLAUDE.md` | Python ML 段补 WF-2 现状一行；强调库常量 `DEFAULT_FUSION_METHOD` / `DEFAULT_MODEL_WEIGHT` 与 `tuning._DEFAULT_RATING_MODE='elo'` 被契约锁定、刻意不改 |
| `README.md` | 新增「建模升级（WF-2）」小节：采纳默认 + 默认关闭能力表，诚实标注未证明 / 回归 / 不可证伪 |
| `docs/data-processing.md` | 流程表 / 术语扩 Elo→Elo+Glicko；校准 / 市场对比补 logit_pool 默认与 opt-in；reliability / calibration 模块说明补 train/serve 配对；新增「WF-2 裁决矩阵」节 + 回归口径声明 |
| `docs/modeling-upgrade-2026-06.md` | 本报告 |

---

## 8. 未来工作

1. **多赛季 Glicko / MOV 再裁决**：当前裁决基于单一回测语料；多赛季历史可重测 Glicko 增益稳定性，并单独裁 MOV 项。
2. **回合分语料补全**：补逐回合 / 比分覆盖后，Glicko-MOV 才从「不可证伪」转为可裁；series BO3 层的逐图自相关假设也能用真实比分校准。
3. **BT / 校准多方法的更大样本 A/B**：no_significant_diff 不等于无效，更大样本可能让 BT 先验 / beta·temperature 校准的小幅差异显著化。
4. **生产 Kelly 上线门槛**：`staking` 仅实现半 Kelly + 敞口上限，实盘前需定下注门槛、最小 edge、组合上限的运营规则。
5. **5E join / odds audit 接线**：把 de-vig 审计字段（overround / devig_z）接进上线前 readiness 审计，并完善 5E 别名 join 的赔率口径核对。
6. **`market_weight` 语义统一**：生产 0.30 与库 / tuning 诊断 0.35 的并存待多赛季后统一。
7. **前端 `swissSim` 同步**：`site/` 的瑞士轮引擎移植自 `swiss.py`，若后续把 Buchholz 配对设为生产默认，需同步前端口径（当前前端解耦、未触碰）。

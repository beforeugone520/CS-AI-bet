# Project Agent Instructions

## Python ML Acceleration

This project supports optional accelerated model backends. Prefer these libraries for training and prediction speed when they are installed:

- `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`

Use the PyPI package name `scikit-learn`; do not install the deprecated `sklearn` package.

The default model stack should use:

- `sklearn.linear_model.LogisticRegression` for the logistic component.
- `sklearn.ensemble.RandomForestClassifier` for the random forest component.
- `xgboost.XGBClassifier` for the xgboost component.
- `joblib` to parallelize independent base-model fitting when practical.

Keep the pure-Python model implementations as a reliable fallback. If an accelerated dependency is missing or fails to import, the code should continue running with the pure-Python backend instead of crashing.

The neural-network component should remain on the existing pure-Python implementation by default. The sklearn MLP path may be enabled only when explicitly requested with `CS2PICKEM_ACCELERATED_MLP=1`, because tiny sample sets can produce noisy numerical warnings.

## 建模升级 WF-2：评级 / 融合默认与 opt-in 清单

WF-2（提交链 WF-2A~2F）在零新依赖、纯 Python 回退、无未来泄漏、train/serve 一致的红线下，给离线建模工具链补齐了一批 SOTA 能力。WF-2F 用「同口径 A/B + paired bootstrap / Diebold-Mariano 显著性」做了大回测裁决，**只翻有证据支持的两个默认**，其余实现但保持默认关闭、仅 opt-in。

### 已采纳的生产默认（显著改进）

1. **赛前评级 `rating_mode='glicko'`（`MatchPredictor.train` 默认）** —— Glicko-2 赛前评级（赛前按周期批量更新，MOV 阻尼 + 不活跃 RD 膨胀，全程无泄漏），写出 `glicko_diff` / `glicko_rd_sum` 候选特征参与 `FeatureSelector`。Elo **仍并存注入**（`inject_elo=True`），glicko 与 elo 两套列同台竞选。WF-2F：在 1301 场 holdout 上 logloss 0.6806 → 0.6700（mean Δ +0.0106，DM p≈0.0007，**显著**）。
2. **odds 融合 `fusion_method='logit_pool'` + 模型权重 0.30** —— 对数意见池（geometric-mean 共识：`logit(p) = w·logit(p_model) + (1−w)·logit(p_market)`，偏市场），在 `forecast`/`pickem` 两条生产路径显式传参。WF-2F：在 515 场赔率子集 walk-forward 上 logloss 0.6371 → 0.5874（对 legacy_clip@0.30 再 +0.0496，DM p=0，**显著**）。

### train/serve 配对铁律（必须同时生效，否则 skew）

`rating_mode='glicko'` 时，`MatchPredictor.train` 在拟合期注入 Glicko 列（`prepare_reliability_features(inject_glicko=True)`），并把整段历史的最终 Glicko 拟合快照进 `team_glicko_state`；serve 端 `predict_probability_details` 检测到非空 `team_glicko_state` 时自动调用 `apply_final_glicko_to_match` 给上场 fixture 注入同口径列。**二者必须成对生效**：只在拟合期注入、scoring 期恒 0，比不注入还差。`team_glicko_state` 为空时该注入是 no-op，因此 `rating_mode='elo'`（opt-in 基线）路径与历史字节一致。

### opt-in 未默认启用清单（默认全部关闭）

| 轴 | 默认值 | WF-2F 裁决 | 如何开启 |
| --- | --- | --- | --- |
| `inject_bt`（Bradley-Terry 实力先验，可与任一 rating_mode 并存） | `False` | no_significant_diff（Δ −0.0008，p≈0.24） | `MatchPredictor.train(inject_bt=True)` |
| `include_unverified`（已抓未用/未验证特征） | `False` | no_significant_diff（Δ −0.0010，p≈0.10） | tuning/特征准备开关 |
| `calibration_method` 非 `'platt'`（`beta` / `temperature`） | `'platt'` | no_significant_diff（三变体 p 均 >0.2） | `train(calibration_method='beta')` |
| `devig_method` 非 `'multiplicative'`（`power` / `shin`） | `'multiplicative'` | **回归**（power Δ −0.0083 p≈0.001、shin Δ −0.0047 p≈0.0003） | 生产 forecast/pickem 路径调 `market_probability_from_row(fixture)`、硬编码 multiplicative，**无生产 opt-in 开关**；仅 `optimize-matches` 诊断 A/B 可对照（`tuning` 的候选轴）。`devig_market(method=...)` 是底层原语，单独调不改变任何生产预测 |
| Glicko-MOV 多赛季再裁决 | MOV 阻尼已实现并随 glicko 启用 | 不可证伪（缺回合分语料覆盖，记为待验证） | 待回合分语料补全后重裁 |
| `pairing='buchholz'`（瑞士轮 Valve 配对）、`series_uplift`（BO3 series 层）、pickem `objective`（`threshold_prob` / `leveraged`）、Kelly 下注（`staking`） | 各自默认沿用历史（`legacy` / 关闭 / `expected_hits` / 不下注） | 实现保留，按需 opt-in | 见 `docs/modeling-upgrade-2026-06.md` |

## Local Python Environment

For this workspace, the acceleration packages were installed into the user Python environment with:

```bash
python3 -m pip install --user --upgrade numpy scikit-learn xgboost joblib
```

On macOS, the installed `xgboost` wheel may require `libomp.dylib`. In this workspace, `libxgboost.dylib` was patched to use the `libomp.dylib` bundled with scikit-learn:

```bash
install_name_tool -change @rpath/libomp.dylib \
  /Users/bruce/Library/Python/3.9/lib/python/site-packages/sklearn/.dylibs/libomp.dylib \
  /Users/bruce/Library/Python/3.9/lib/python/site-packages/xgboost/lib/libxgboost.dylib
```

Before relying on accelerated training, verify the active backends:

```bash
PYTHONPATH=src python3 - <<'PY'
from cs2pickem.models import default_ensemble
print(default_ensemble(seed=7, epochs=2, n_jobs=1).component_backends)
PY
```

Expected default output after acceleration is available:

```text
{'logistic': 'sklearn', 'random_forest': 'sklearn', 'xgboost': 'xgboost', 'neural_network': 'pure_python'}
```

## Verification

After changing model code or dependency behavior, run（全量 512 项应保持全绿；WF-2 翻转后无数值基线测试需更新、行为契约测试全保持）：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

For a quick runtime smoke test, run:

```bash
PYTHONPATH=src python3 -m cs2pickem.cli demo
```

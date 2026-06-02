"""Chronological out-of-sample backtest: NN-fixed 4-model vs no-NN 3-model vs baselines."""
import csv, json, sys
sys.path.insert(0, "src")
from cs2pickem.predictor import MatchPredictor
from cs2pickem.evaluation import accuracy, log_loss, auc, brier_score, calibration_table

CORPUS = "data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/enriched_matches.csv"
OUT = "data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/backtest_configs_2026-06-01.json"

rows = list(csv.DictReader(open(CORPUS)))
rows = [r for r in rows if r.get("date") and r.get("winner") and r.get("team1") and r.get("team2")]
rows.sort(key=lambda r: r["date"])
cut = int(len(rows) * 0.8)
train, test = rows[:cut], rows[cut:]
split_date = train[-1]["date"]
print(f"corpus={len(rows)} train={len(train)} test={len(test)} split_date={split_date}", flush=True)

def labels_of(rs):
    return [1 if r["winner"] == r["team1"] else 0 for r in rs]
y = labels_of(test)

def metrics(name, probs):
    cal = calibration_table(y, probs, bins=10)
    m = {"config": name, "n": len(y),
         "accuracy": round(accuracy(y, probs), 4), "auc": round(auc(y, probs), 4),
         "log_loss": round(log_loss(y, probs), 4), "brier": round(brier_score(y, probs), 4),
         "ece": round(cal["ece"], 4)}
    print(f"  {name:<22} acc={m['accuracy']} auc={m['auc']} logloss={m['log_loss']} brier={m['brier']} ece={m['ece']}", flush=True)
    return m

def num(v, d=0.5):
    try: return float(v)
    except (TypeError, ValueError): return d

results = []
# --- Baselines (no training) ---
print("BASELINES", flush=True)
results.append(metrics("baseline_0.5", [0.5] * len(test)))
form = []
for r in test:
    w1, w2 = num(r.get("team1_recent_winrate_10")), num(r.get("team2_recent_winrate_10"))
    form.append(w1 / (w1 + w2) if (w1 + w2) > 0 else 0.5)
results.append(metrics("baseline_recent_form", form))

# --- Model configs ---
CONFIGS = {
    "model_4_nn_fixed": {"logistic": 0.20, "neural_network": 0.15, "random_forest": 0.30, "xgboost": 0.35},
    "model_3_no_nn":    {"logistic": 0.20, "neural_network": 0.0,  "random_forest": 0.30, "xgboost": 0.35},
}
for name, weights in CONFIGS.items():
    print(f"TRAIN {name}", flush=True)
    predictor = MatchPredictor.train(train, reference_date=split_date, top_k=25,
                                     max_age_days=400, ensemble_weights=weights)
    probs = [max(1e-6, min(1 - 1e-6, predictor.predict_with_maps(r, None)[0])) for r in test]
    results.append(metrics(name, probs))

best = min(results, key=lambda m: m["log_loss"])
print(f"\nBEST by log_loss: {best['config']}", flush=True)
json.dump({"split_date": split_date, "train_n": len(train), "test_n": len(test),
           "results": results, "best_by_log_loss": best["config"]},
          open(OUT, "w"), indent=2)
print(f"written: {OUT}", flush=True)
print("ALL DONE", flush=True)

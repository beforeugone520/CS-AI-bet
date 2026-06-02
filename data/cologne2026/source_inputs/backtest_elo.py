"""Does the corpus-derived Elo feature improve out-of-sample prediction?

Computes leakage-free pre-match Elo over the full corpus, injects team1_elo/team2_elo,
then trains the best config (3-model, no NN) WITH vs WITHOUT Elo on the same
chronological split. Also writes final per-team Elo for seeding the Cologne fixtures.
"""
import csv, json, sys
sys.path.insert(0, "src")
from cs2pickem.predictor import MatchPredictor
from cs2pickem.ratings import compute_elo_ratings
from cs2pickem.evaluation import accuracy, log_loss, auc, brier_score, calibration_table

CORPUS = "data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/enriched_matches.csv"
OUT = "data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/backtest_elo_2026-06-01.json"
ELO_OUT = "data/cologne2026/predictions/fivee_6m_stage1_2026-06-01/team_elo_2026-06-01.json"
WEIGHTS = {"logistic": 0.20, "neural_network": 0.0, "random_forest": 0.30, "xgboost": 0.35}

rows = list(csv.DictReader(open(CORPUS)))
rows = [r for r in rows if r.get("date") and r.get("winner") and r.get("team1") and r.get("team2")]
rows.sort(key=lambda r: r["date"])

# Leakage-free pre-match Elo over the full corpus (tier-weighted: Major/S move more).
per_match, final_elo = compute_elo_ratings(
    rows, base=1500.0, k=24.0, tier_k={"S": 32.0, "A": 20.0, "B": 14.0, "C": 10.0})
for r, e in zip(rows, per_match):
    r["team1_elo"] = e["team1_elo_pre"]
    r["team2_elo"] = e["team2_elo_pre"]

cut = int(len(rows) * 0.8)
train, test = rows[:cut], rows[cut:]
split_date = train[-1]["date"]
y = [1 if r["winner"] == r["team1"] else 0 for r in test]
print(f"corpus={len(rows)} train={len(train)} test={len(test)} split={split_date} teams_rated={len(final_elo)}", flush=True)

def metrics(name, probs):
    cal = calibration_table(y, probs, bins=10)
    m = {"config": name, "accuracy": round(accuracy(y, probs), 4), "auc": round(auc(y, probs), 4),
         "log_loss": round(log_loss(y, probs), 4), "brier": round(brier_score(y, probs), 4),
         "ece": round(cal["ece"], 4)}
    print(f"  {name:<18} acc={m['accuracy']} auc={m['auc']} logloss={m['log_loss']} brier={m['brier']} ece={m['ece']}", flush=True)
    return m

results = []
for name, with_elo in [("no_elo", False), ("with_elo", True)]:
    print(f"TRAIN {name}", flush=True)
    tr = [dict(r) for r in train]
    te = [dict(r) for r in test]
    if not with_elo:  # neutralize Elo -> elo_diff = 0 (isolates the feature's effect)
        for r in tr + te:
            r["team1_elo"] = 1500.0
            r["team2_elo"] = 1500.0
    predictor = MatchPredictor.train(tr, reference_date=split_date, top_k=25,
                                     max_age_days=400, ensemble_weights=WEIGHTS)
    elo_selected = "elo_diff" in predictor.selected_feature_names
    probs = [max(1e-6, min(1 - 1e-6, predictor.predict_with_maps(r, None)[0])) for r in te]
    m = metrics(name, probs)
    m["elo_diff_selected"] = elo_selected
    print(f"     elo_diff selected as feature: {elo_selected}", flush=True)
    results.append(m)

best = min(results, key=lambda m: m["log_loss"])
print(f"\nBEST by log_loss: {best['config']}", flush=True)
json.dump({"split_date": split_date, "results": results, "best": best["config"]}, open(OUT, "w"), indent=2)
json.dump({"as_of": "2026-06-01", "base": 1500.0, "ratings": final_elo}, open(ELO_OUT, "w"), indent=2)
print(f"written: {OUT}\nwritten: {ELO_OUT}", flush=True)
print("ALL DONE", flush=True)

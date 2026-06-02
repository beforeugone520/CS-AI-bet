"""Augmented Swiss Pick'em: retrain on 5E + fetched data, with team form refreshed
from the augmented corpus, then run 100k Monte Carlo. Does the model ALONE (given data)
now produce a sensible 3-0/advance/0-3 that matches experts/market?"""
import csv, json, sys
sys.path.insert(0, "src")
from cs2pickem.enrichment import enrich_match_history
from cs2pickem.pickem import model_driven_pickems

BASE = "data/cologne2026"
raw = list(csv.DictReader(open(f"{BASE}/processed/fivee_training_history_maps_6m_2026-06-01.csv")))
fetched = list(csv.DictReader(open(f"{BASE}/source_inputs/missing_teams_results_2026-06-01.csv")))
for r in fetched:
    r.setdefault("map", "unknown")
combined = raw + fetched
print(f"combined={len(combined)}; enriching...", flush=True)
enriched = enrich_match_history(combined)

# Current form per team from most recent enriched appearance.
FORM = ["recent_winrate_10", "bo1_winrate_6m", "bo3_winrate_6m"]
current = {}
for row in enriched:
    for side in ("team1", "team2"):
        current[row[side]] = {f: row.get(f"{side}_{f}") for f in FORM}

teams = list(csv.DictReader(open(f"{BASE}/processed/stage1_teams_prediction_fivee_6m_model_2026-06-01.csv")))
refreshed = 0
for trow in teams:
    cur = current.get(trow["team"])
    if cur:
        for f in FORM:
            if cur.get(f) not in (None, ""):
                trow[f] = cur[f]
        refreshed += 1
print(f"teams refreshed from augmented corpus: {refreshed}/{len(teams)}", flush=True)

print("training + 100k Swiss Monte Carlo (3-model no NN)...", flush=True)
report = model_driven_pickems(
    enriched, teams, reference_date="2026-06-01", simulations=100000,
    top_k=25, stage="challengers", max_age_days=400,
    ensemble_weights={"logistic": 0.20, "neural_network": 0.0, "random_forest": 0.30, "xgboost": 0.35})
OUT = f"{BASE}/predictions/fivee_6m_stage1_2026-06-01/pickem_augmented_2026-06-01.json"
json.dump(report, open(OUT, "w"), indent=2, default=str)

tp = report["team_probabilities"]
pk = report["pickems"]
print("\n=== AUGMENTED model-only Pick'em ===", flush=True)
print("3-0:", pk["3-0"], flush=True)
print("advance:", pk["advance"], flush=True)
print("0-3:", pk["0-3"], flush=True)
print("\n=== team outcome probs (sorted by advance) ===", flush=True)
for t in sorted(tp, key=lambda x: tp[x]["advance"], reverse=True):
    print(f'  {t:<20} 3-0={tp[t]["3-0"]:.3f} adv={tp[t]["advance"]:.3f} 0-3={tp[t]["0-3"]:.3f}', flush=True)
print(f"\nwritten: {OUT}", flush=True)
print("ALL DONE", flush=True)

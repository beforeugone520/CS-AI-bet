"""Integration validation: add the 69 fetched Western-team matches to the corpus,
re-enrich (with Elo), rebuild the Cologne fixtures with real form+Elo for all teams,
retrain (3-model no-NN), and re-forecast. Does the model self-correct GamerLegion?"""
import csv, json, sys
sys.path.insert(0, "src")
from cs2pickem.enrichment import enrich_match_history
from cs2pickem.ratings import compute_elo_ratings
from cs2pickem.forecast import forecast_fixtures

BASE = "data/cologne2026"
RAW_5E = f"{BASE}/processed/fivee_training_history_maps_6m_2026-06-01.csv"
FETCHED = f"{BASE}/source_inputs/missing_teams_results_2026-06-01.csv"
FIXTURES = f"{BASE}/processed/stage1_opening_fixtures_fivee_6m_model_2026-06-01.csv"
OLD_FC = f"{BASE}/predictions/fivee_6m_stage1_2026-06-01/forecast_report.json"
OUT = f"{BASE}/predictions/fivee_6m_stage1_2026-06-01/forecast_augmented_2026-06-01.json"
WEIGHTS = {"logistic": 0.20, "neural_network": 0.0, "random_forest": 0.30, "xgboost": 0.35}

raw = list(csv.DictReader(open(RAW_5E)))
fetched = list(csv.DictReader(open(FETCHED)))
for r in fetched:
    r.setdefault("map", "unknown")
combined = raw + fetched
print(f"raw_5e={len(raw)} fetched={len(fetched)} combined={len(combined)}", flush=True)

print("enriching (with Elo)...", flush=True)
enriched = enrich_match_history(combined)
_, final_elo = compute_elo_ratings(combined, base=1500.0, k=24.0,
                                   tier_k={"S": 32.0, "A": 20.0, "B": 14.0, "C": 10.0})

# Current form per team = features from their most recent enriched appearance.
FORM = ["recent_winrate_5", "recent_winrate_10", "bo1_winrate_6m", "bo3_winrate_6m",
        "map_winrate", "matches_30d", "current_streak"]
current = {}
for row in enriched:  # chronological; later rows overwrite -> ends on most recent
    for side in ("team1", "team2"):
        t = row[side]
        current[t] = {f: row.get(f"{side}_{f}") for f in FORM}

def feat(team, f, default):
    v = (current.get(team) or {}).get(f)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

# Rebuild the 8 opening fixtures with augmented form + Elo (keep static/swiss fields).
fixtures = list(csv.DictReader(open(FIXTURES)))
for fx in fixtures:
    for side in ("team1", "team2"):
        t = fx[side]
        fx[f"{side}_elo"] = final_elo.get(t, 1500.0)
        fx[f"{side}_recent_winrate_10"] = feat(t, "recent_winrate_10", 0.5)
        fx[f"{side}_recent_winrate_5"] = feat(t, "recent_winrate_5", 0.5)
        fx[f"{side}_bo1_winrate_6m"] = feat(t, "bo1_winrate_6m", 0.5)
        fx[f"{side}_bo3_winrate_6m"] = feat(t, "bo3_winrate_6m", 0.5)
        fx[f"{side}_matches_30d"] = feat(t, "matches_30d", 0)

print("training 3-model (no NN) + forecasting...", flush=True)
report = forecast_fixtures(enriched, fixtures, reference_date="2026-06-01",
                           top_k=25, max_age_days=400, ensemble_weights=WEIGHTS)
json.dump(report, open(OUT, "w"), indent=2, default=str)

old = {p["team1"]: p["model_probability_team1"] for p in json.load(open(OLD_FC))["predictions"]}
print("\n=== Cologne Elo (augmented corpus) for the 4 formerly-missing teams ===", flush=True)
for t in ["GamerLegion", "BetBoom", "HEROIC", "Lynn Vision"]:
    print(f"  {t:<14} elo={final_elo.get(t,1500):.0f}  recent_wr10={feat(t,'recent_winrate_10',0.5):.2f}", flush=True)

print("\n=== forecast: OLD (5E only) vs NEW (augmented+Elo), P(team1 win) ===", flush=True)
print(f'{"matchup":<34}{"old":>8}{"new":>8}  pick', flush=True)
for p in report["predictions"]:
    t1 = p["team1"]
    print(f'{t1+" vs "+p["team2"]:<34}{old.get(t1,float("nan")):>8.3f}{p["model_probability_team1"]:>8.3f}  -> {p["pick"]}', flush=True)
print(f"\nelo_diff selected: {'elo_diff' in report['selected_feature_names']}", flush=True)
print(f"written: {OUT}", flush=True)
print("ALL DONE", flush=True)

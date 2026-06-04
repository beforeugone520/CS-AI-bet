from __future__ import annotations

import argparse
import json

from .backtest import (
    backtest_forecast_file,
    backtest_pickem_file,
    backtest_pickem_suite_file,
    checkpoint_pickem_file,
    replay_pickem_backtest_suite_file,
)
from .bp import merge_bp_file
from .data import read_matches_csv, read_teams_csv, write_json
from .export import build_pickem_answer_sheet_file
from .fivee import collect_fivee_match_results, collect_fivee_team_pages, read_urls
from .forecast import apply_forecast_policy_file, forecast_fixtures_file
from .odds import merge_odds_file
from .pickem import model_driven_pickems_file
from .pipeline import enrich_matches_file, run_demo, simulate_from_team_rows, train_evaluate
from .players import merge_player_stats_file
from .readiness import DEFAULT_PICKEM_SLOTS, audit_readiness_file
from .tuning import optimize_match_predictions
from .update import (
    daily_update_from_config,
    update_dataset_from_html,
    update_dataset_from_url,
    update_event_teams_from_html,
    update_event_teams_from_url,
    update_player_stats_from_html,
    update_player_stats_from_url,
    update_rankings_from_html,
    update_rankings_from_url,
)
from .visualization import visualize_training_report_file
from .workflow import run_end_to_end_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="CS2 Major Pick'em ML offline toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="run the built-in offline demo")
    train_parser = subparsers.add_parser("train", help="train/evaluate from a match CSV")
    train_parser.add_argument("--matches", required=True, help="path to match CSV")
    train_parser.add_argument("--reference-date", required=True, help="YYYY-MM-DD date for freshness filtering")
    train_parser.add_argument("--top-k", type=int, default=25, help="number of selected features to keep")
    train_parser.add_argument("--cv-folds", type=int, default=5, help="number of chronological CV folds")
    train_parser.add_argument("--train-ratio", type=float, default=0.8, help="chronological training split ratio")
    train_parser.add_argument("--validation-ratio", type=float, default=0.1, help="chronological validation split ratio")
    train_parser.add_argument("--max-age-days", type=int, default=90, help="freshness window for training rows; use 180 for six-month Cologne prep")
    train_parser.add_argument("--train-end-date", help="optional calendar split boundary, e.g. 2026-04-30")
    train_parser.add_argument("--validation-end-date", help="optional calendar split boundary, e.g. 2026-05-15")
    train_parser.add_argument("--output", help="optional JSON output path")
    tune_parser = subparsers.add_parser("optimize-matches", help="replay historical matches and tune model configuration on a validation split")
    tune_parser.add_argument("--matches", required=True, help="historical enriched/training match CSV")
    tune_parser.add_argument("--reference-date", required=True, help="YYYY-MM-DD date for training freshness filtering")
    tune_parser.add_argument("--train-ratio", type=float, default=0.8, help="chronological training split ratio")
    tune_parser.add_argument("--validation-ratio", type=float, default=0.1, help="chronological validation split ratio")
    tune_parser.add_argument("--max-age-days", type=int, default=400, help="freshness window for historical replay rows")
    tune_parser.add_argument("--top-k-values", default="12,18,25", help="comma-separated selected-feature counts to search")
    tune_parser.add_argument("--epochs-values", default="8", help="comma-separated training epoch counts to search")
    tune_parser.add_argument("--candidates", default="fast_logistic,random_forest,no_nn", help="comma-separated candidate presets: fast_logistic,logistic,random_forest,xgboost,default,no_nn,tree_blend")
    tune_parser.add_argument("--seed", type=int, default=29)
    tune_parser.add_argument("--no-calibration", action="store_true", help="disable validation-fitted probability calibration for test metrics")
    tune_parser.add_argument("--rolling-folds", type=int, default=3, help="number of chronological rolling validation folds to score each candidate")
    tune_parser.add_argument("--market-weight", type=float, default=0.30, help="market probability weight for historical model+market fusion diagnostics")
    tune_parser.add_argument("--probability-objective", default="log_loss", choices=["accuracy", "log_loss", "brier_score", "ece"], help="objective used to choose raw vs calibrated test probabilities")
    tune_parser.add_argument("--elo-modes", default="with", help="comma-separated Elo feature modes to compare: with,without")
    tune_parser.add_argument("--output", help="optional JSON output path")
    simulate_parser = subparsers.add_parser("simulate", help="simulate Swiss from a team CSV")
    simulate_parser.add_argument("--teams", required=True, help="CSV with team,seed,strength columns")
    simulate_parser.add_argument("--simulations", type=int, default=100000)
    simulate_parser.add_argument("--seed", type=int, default=13)
    simulate_parser.add_argument("--output", help="optional JSON output path")
    pickem_parser = subparsers.add_parser("pickem", help="train a match model, run Swiss Monte Carlo, and output Pick'em picks")
    pickem_parser.add_argument("--history", required=True, help="historical enriched/training match CSV")
    pickem_parser.add_argument("--teams", required=True, help="CSV with team seed and pre-event team features")
    pickem_parser.add_argument("--fixtures", help="optional upcoming fixture CSV with matchup-level odds for Swiss simulation")
    pickem_parser.add_argument("--reference-date", required=True, help="YYYY-MM-DD date for training freshness filtering")
    pickem_parser.add_argument("--profiles", help="optional team profiles JSON for unknown map averaging")
    pickem_parser.add_argument("--simulations", type=int, default=100000)
    pickem_parser.add_argument("--seed", type=int, default=13)
    pickem_parser.add_argument("--top-k", type=int, default=25)
    pickem_parser.add_argument("--epochs", type=int, default=50)
    pickem_parser.add_argument("--stage", default="default", choices=["default", "challengers", "legends"], help="stage-specific Pick'em risk profile")
    pickem_parser.add_argument("--max-age-days", type=int, default=90, help="freshness window for training rows")
    pickem_parser.add_argument("--output", help="optional JSON output path")
    answer_sheet_parser = subparsers.add_parser("answer-sheet", help="export a compact final Pick'em answer sheet from pickem/readiness reports")
    answer_sheet_parser.add_argument("--pickem-report", required=True, help="JSON output from pickem or pipeline command")
    answer_sheet_parser.add_argument("--readiness-report", help="optional JSON output from readiness or pipeline command")
    answer_sheet_parser.add_argument("--minimum-selection-margin", type=float, default=0.04, help="warning threshold for narrow Pick'em selection margins")
    answer_sheet_parser.add_argument("--output", help="optional JSON output path")
    backtest_parser = subparsers.add_parser("backtest-pickem", help="score Pick'em picks against final Swiss standings")
    backtest_parser.add_argument("--pickems", required=True, help="pickem report JSON or raw pickems JSON")
    backtest_parser.add_argument("--results", required=True, help="CSV with team,wins,losses final Swiss standings")
    backtest_parser.add_argument("--pass-threshold", type=int, default=5, help="minimum correct picks considered a pass")
    backtest_parser.add_argument("--output", help="optional JSON output path")
    checkpoint_parser = subparsers.add_parser(
        "checkpoint-pickem",
        help="classify Pick'em slots as locked, alive, or broken against current Swiss standings",
    )
    checkpoint_parser.add_argument("--pickems", required=True, help="pickem report JSON or raw pickems JSON")
    checkpoint_parser.add_argument("--standings", required=True, help="CSV with team,wins,losses current Swiss standings")
    checkpoint_parser.add_argument("--output", help="optional JSON output path")
    forecast_backtest_parser = subparsers.add_parser(
        "backtest-forecast",
        help="score single-match forecast report against actual match results",
    )
    forecast_backtest_parser.add_argument("--forecast-report", required=True, help="JSON output from forecast command")
    forecast_backtest_parser.add_argument("--results", required=True, help="CSV with date,team1,team2,winner actual results")
    forecast_backtest_parser.add_argument("--output", help="optional JSON output path")
    backtest_suite_parser = subparsers.add_parser("backtest-pickem-suite", help="score multiple historical Pick'em cases and report pass rate")
    backtest_suite_parser.add_argument("--suite", required=True, help="JSON list or {cases: [...]} with pickems/results or pickems_path/results_path")
    backtest_suite_parser.add_argument("--pass-threshold", type=int, default=5, help="minimum correct picks considered a case pass")
    backtest_suite_parser.add_argument("--pass-rate-target", type=float, default=0.38, help="historical suite pass-rate target")
    backtest_suite_parser.add_argument("--output", help="optional JSON output path")
    replay_suite_parser = subparsers.add_parser("replay-pickem-suite", help="retrain, regenerate, and score historical Pick'em replay cases")
    replay_suite_parser.add_argument("--suite", required=True, help="JSON list or {cases: [...]} with history/teams/results paths or inline rows")
    replay_suite_parser.add_argument("--pass-threshold", type=int, default=5, help="minimum correct picks considered a case pass")
    replay_suite_parser.add_argument("--pass-rate-target", type=float, default=0.38, help="historical replay pass-rate target")
    replay_suite_parser.add_argument("--simulations", type=int, default=100000, help="default Swiss simulations per replay case")
    replay_suite_parser.add_argument("--top-k", type=int, default=25, help="default selected feature count per replay case")
    replay_suite_parser.add_argument("--epochs", type=int, default=50, help="default training epochs per replay case")
    replay_suite_parser.add_argument("--max-age-days", type=int, default=90, help="default freshness window per replay case")
    replay_suite_parser.add_argument("--output", help="optional JSON output path")
    enrich_parser = subparsers.add_parser("enrich", help="build rolling training features from raw match CSV")
    enrich_parser.add_argument("--matches", required=True, help="raw chronological or unsorted match CSV")
    enrich_parser.add_argument("--output", required=True, help="path for enriched match CSV")
    enrich_parser.add_argument("--profiles-output", help="optional team map-profile JSON output path")
    forecast_parser = subparsers.add_parser("forecast", help="train on history CSV and predict upcoming fixture CSV")
    forecast_parser.add_argument("--history", required=True, help="historical enriched/training match CSV")
    forecast_parser.add_argument("--fixtures", required=True, help="upcoming fixture CSV")
    forecast_parser.add_argument("--reference-date", required=True, help="YYYY-MM-DD date for training freshness filtering")
    forecast_parser.add_argument("--profiles", help="optional team profiles JSON for unknown map averaging")
    forecast_parser.add_argument("--bp", help="optional BP intel CSV to override confirmed maps/bans")
    forecast_parser.add_argument("--top-k", type=int, default=25, help="number of selected features to keep")
    forecast_parser.add_argument("--epochs", type=int, default=50, help="training epochs for lightweight learners")
    forecast_parser.add_argument("--max-age-days", type=int, default=90, help="freshness window for training rows")
    forecast_parser.add_argument("--minimum-margin", type=float, default=0.02, help="minimum probability margin above 50%% required for an actionable single-match pick")
    forecast_parser.add_argument("--avoid-player-form-counter-signal", action="store_true", help="avoid actionable picks when short-term player form points against the predicted side")
    forecast_parser.add_argument("--player-form-counter-min-confidence", type=float, default=0.4, help="minimum player form sample confidence required for counter-signal avoidance")
    forecast_parser.add_argument("--output", help="optional JSON output path")
    apply_forecast_policy_parser = subparsers.add_parser("apply-forecast-policy", help="apply single-match decision policy to an existing forecast report without retraining")
    apply_forecast_policy_parser.add_argument("--forecast-report", required=True, help="JSON output from forecast command")
    apply_forecast_policy_parser.add_argument("--fixtures", help="optional fixture CSV with player form fields to merge into predictions")
    apply_forecast_policy_parser.add_argument("--minimum-margin", type=float, default=0.02, help="minimum probability margin above 50%% required for an actionable single-match pick")
    apply_forecast_policy_parser.add_argument("--avoid-player-form-counter-signal", action="store_true", help="avoid actionable picks when short-term player form points against the predicted side")
    apply_forecast_policy_parser.add_argument("--player-form-counter-min-confidence", type=float, default=0.4, help="minimum player form sample confidence required for counter-signal avoidance")
    apply_forecast_policy_parser.add_argument("--output", help="optional JSON output path")
    odds_parser = subparsers.add_parser("merge-odds", help="merge multi-provider decimal odds into match/fixture CSV")
    odds_parser.add_argument("--matches", required=True, help="match or fixture CSV")
    odds_parser.add_argument("--odds", required=True, help="odds CSV with date,provider,team1,team2,odds_team1,odds_team2")
    odds_parser.add_argument("--output", required=True, help="augmented CSV output path")
    players_parser = subparsers.add_parser("merge-players", help="merge recent player stats into match/fixture CSV")
    players_parser.add_argument("--matches", required=True, help="match or fixture CSV")
    players_parser.add_argument("--players", required=True, help="player stats CSV with date,team,player,rating,kd,opening_success,clutch_winrate,is_substitute")
    players_parser.add_argument("--output", required=True, help="augmented CSV output path")
    players_parser.add_argument("--window-days", type=int, default=15, help="lookback window for prior player rows")
    bp_parser = subparsers.add_parser("merge-bp", help="merge map veto/BP intel into fixture CSV")
    bp_parser.add_argument("--fixtures", required=True, help="upcoming fixture CSV")
    bp_parser.add_argument("--bp", required=True, help="BP intel CSV with date,team1,team2,map/source/confidence")
    bp_parser.add_argument("--output", required=True, help="augmented fixture CSV output path")
    readiness_parser = subparsers.add_parser("readiness", help="audit whether data and training metrics satisfy production Pick'em gates")
    readiness_parser.add_argument("--matches", required=True, help="training match CSV")
    readiness_parser.add_argument("--training-report", required=True, help="JSON output from train command")
    readiness_parser.add_argument("--minimum-rows", type=int, default=8000)
    readiness_parser.add_argument("--required-teams", type=int, default=80)
    readiness_parser.add_argument("--participants", help="CSV with declared Major participant teams")
    readiness_parser.add_argument("--top-teams", help="CSV with Top80 or scoped professional teams")
    readiness_parser.add_argument("--expected-train-end-date", help="require training report to use this calendar train boundary")
    readiness_parser.add_argument("--expected-validation-end-date", help="require training report to use this calendar validation boundary")
    readiness_parser.add_argument("--minimum-max-age-days", type=int, help="require training report freshness window to be at least this many days")
    readiness_parser.add_argument("--sample-reference-date", help="reference date for row-level sample freshness scope")
    readiness_parser.add_argument("--maximum-sample-age-days", type=int, default=180, help="maximum row age allowed by readiness data-quality scope")
    readiness_parser.add_argument("--pickem-backtest-report", help="JSON output from backtest-pickem-suite")
    readiness_parser.add_argument("--pickem-pass-rate-target", type=float, default=0.38, help="minimum historical Pick'em suite pass rate")
    readiness_parser.add_argument("--pickem-report", help="JSON output from pickem or pipeline command")
    readiness_parser.add_argument("--forecast-report", help="JSON output from forecast or pipeline command")
    readiness_parser.add_argument("--minimum-pickem-simulations", type=int, help="require Pick'em Swiss Monte Carlo simulations to be at least this count")
    readiness_parser.add_argument("--require-pickem-slots", action="store_true", help="require complete 3-0/advance/0-3 Pick'em answer slots")
    readiness_parser.add_argument("--minimum-pickem-selection-margin", type=float, help="require every Pick'em detail selection margin to be at least this value")
    readiness_parser.add_argument("--minimum-pickem-market-adjusted-matchups", type=int, help="require Pick'em Swiss matchup probabilities to use real market odds at least this many times")
    readiness_parser.add_argument("--require-forecast-low-confidence-avoidance", action="store_true", help="require low-confidence forecast fixtures to use the avoid pick")
    readiness_parser.add_argument("--source-manifest", action="append", dest="source_manifests", help="source/update manifest JSON to require as freshly refreshed")
    readiness_parser.add_argument("--required-source", action="append", dest="required_sources", help="source name that must appear in the supplied source manifests")
    readiness_parser.add_argument("--source-reference-time", help="ISO timestamp used to evaluate source freshness; defaults to current UTC")
    readiness_parser.add_argument("--maximum-source-age-hours", type=int, help="maximum allowed age for each supplied source manifest; defaults to 24 when source manifests are supplied")
    readiness_parser.add_argument("--require-validation-tuned-weights", action="store_true", help="require validation log-loss tuned ensemble weights in the training report")
    readiness_parser.add_argument("--output", help="optional JSON output path")
    visualize_parser = subparsers.add_parser("visualize", help="render training report feature and probability charts")
    visualize_parser.add_argument("--training-report", required=True, help="JSON output from train command")
    visualize_parser.add_argument("--output-dir", required=True, help="directory for chart files and manifest")
    visualize_parser.add_argument("--prefix", default="training", help="file prefix for generated charts")
    pipeline_parser = subparsers.add_parser("pipeline", help="run the offline end-to-end Pick'em workflow")
    pipeline_parser.add_argument("--history", required=True, help="raw historical match CSV")
    pipeline_parser.add_argument("--fixtures", required=True, help="upcoming fixture CSV")
    pipeline_parser.add_argument("--teams", required=True, help="team seed/profile CSV")
    pipeline_parser.add_argument("--reference-date", required=True, help="YYYY-MM-DD date for freshness filtering")
    pipeline_parser.add_argument("--output-dir", required=True, help="directory for all pipeline artifacts")
    pipeline_parser.add_argument("--odds", help="optional odds feed CSV")
    pipeline_parser.add_argument("--players", help="optional player stats CSV")
    pipeline_parser.add_argument("--bp", help="optional BP intel CSV")
    pipeline_parser.add_argument("--simulations", type=int, default=100000)
    pipeline_parser.add_argument("--top-k", type=int, default=25)
    pipeline_parser.add_argument("--epochs", type=int, default=50)
    pipeline_parser.add_argument("--stage", default="default", choices=["default", "challengers", "legends"], help="stage-specific Pick'em risk profile")
    pipeline_parser.add_argument("--cv-folds", type=int, default=5)
    pipeline_parser.add_argument("--window-days", type=int, default=15)
    pipeline_parser.add_argument("--max-age-days", type=int, default=90, help="freshness window for train/forecast/pickem model training")
    pipeline_parser.add_argument("--train-end-date", help="optional calendar split boundary for train report")
    pipeline_parser.add_argument("--validation-end-date", help="optional calendar split boundary for train report")
    pipeline_parser.add_argument("--version-log", help="optional date,version_tag log to annotate training history before readiness")
    pipeline_parser.add_argument("--minimum-rows", type=int, default=8000)
    pipeline_parser.add_argument("--required-teams", type=int, default=80)
    pipeline_parser.add_argument("--participants", help="CSV with declared Major participant teams")
    pipeline_parser.add_argument("--top-teams", help="CSV with Top80 or scoped professional teams")
    pipeline_parser.add_argument("--pickem-backtest-report", help="JSON output from backtest-pickem-suite used as a production readiness gate")
    pipeline_parser.add_argument("--pickem-pass-rate-target", type=float, default=0.38, help="minimum historical Pick'em suite pass rate for pipeline readiness")
    pipeline_parser.add_argument("--minimum-pickem-simulations", type=int, default=100000, help="minimum Swiss Monte Carlo simulations required by pipeline readiness")
    pipeline_parser.add_argument("--skip-pickem-slot-check", action="store_true", help="do not require complete Pick'em answer slots in pipeline readiness")
    pipeline_parser.add_argument("--minimum-pickem-selection-margin", type=float, default=0.04, help="minimum Pick'em selection margin required by pipeline readiness")
    pipeline_parser.add_argument("--minimum-pickem-market-adjusted-matchups", type=int, help="minimum Pick'em Swiss matchup odds adjustments required by pipeline readiness; defaults to 1 when --odds is supplied")
    pipeline_parser.add_argument("--source-manifest", action="append", dest="source_manifests", help="source/update manifest JSON to require as freshly refreshed in pipeline readiness")
    pipeline_parser.add_argument("--required-source", action="append", dest="required_sources", help="source name that must appear in the supplied source manifests")
    pipeline_parser.add_argument("--source-reference-time", help="ISO timestamp used to evaluate source freshness; defaults to current UTC")
    pipeline_parser.add_argument("--maximum-source-age-hours", type=int, help="maximum allowed age for each supplied source manifest; defaults to 24 when source manifests are supplied")
    update_parser = subparsers.add_parser("update", help="parse cached/source HTML into a match JSON dataset")
    update_input = update_parser.add_mutually_exclusive_group(required=True)
    update_input.add_argument("--html", help="path to saved HLTV-like result HTML")
    update_input.add_argument("--url", help="HLTV-like results URL to fetch through cache")
    update_parser.add_argument("--cache-dir", default=".cs2pickem-cache", help="HTML cache directory for --url")
    update_parser.add_argument("--refresh", action="store_true", help="ignore cached HTML for --url")
    update_parser.add_argument("--output", required=True, help="path for parsed match JSON")
    update_parser.add_argument("--manifest", required=True, help="path for update manifest JSON")
    update_parser.add_argument("--version-log", help="CSV/text file with date,version_tag rows")
    update_parser.add_argument("--source-name", default="hltv")
    update_parser.add_argument("--dataset", help="optional long-lived training CSV to append parsed rows into")
    update_parser.add_argument("--dataset-manifest", help="optional manifest JSON for --dataset")
    update_parser.add_argument("--team-metadata", help="optional team metadata CSV to fill rank/RMR/Major fields before append")
    update_parser.add_argument("--players", help="optional player stats CSV to fill team player fields before append")
    update_parser.add_argument("--window-days", type=int, default=15, help="lookback window for --players")
    update_parser.add_argument("--default-swiss-state", action="store_true", help="fill neutral Swiss round/win/loss state for parsed results")
    event_teams_parser = subparsers.add_parser("event-teams", help="parse HLTV-like event team pages into readiness team CSV")
    event_teams_input = event_teams_parser.add_mutually_exclusive_group(required=True)
    event_teams_input.add_argument("--html", help="path to saved HLTV-like event HTML")
    event_teams_input.add_argument("--url", help="HLTV-like event URL to fetch through cache")
    event_teams_parser.add_argument("--cache-dir", default=".cs2pickem-cache", help="HTML cache directory for --url")
    event_teams_parser.add_argument("--refresh", action="store_true", help="ignore cached HTML for --url")
    event_teams_parser.add_argument("--output", required=True, help="participant/team CSV output path")
    event_teams_parser.add_argument("--manifest", required=True, help="path for event-team manifest JSON")
    event_teams_parser.add_argument("--source-name", default="hltv-event")
    rankings_parser = subparsers.add_parser("rankings", help="parse HLTV-like ranking pages into Top-N team CSV")
    rankings_input = rankings_parser.add_mutually_exclusive_group(required=True)
    rankings_input.add_argument("--html", help="path to saved HLTV-like ranking HTML")
    rankings_input.add_argument("--url", help="HLTV-like ranking URL to fetch through cache")
    rankings_parser.add_argument("--cache-dir", default=".cs2pickem-cache", help="HTML cache directory for --url")
    rankings_parser.add_argument("--refresh", action="store_true", help="ignore cached HTML for --url")
    rankings_parser.add_argument("--output", required=True, help="Top-N team CSV output path")
    rankings_parser.add_argument("--manifest", required=True, help="path for ranking manifest JSON")
    rankings_parser.add_argument("--limit", type=int, default=80)
    rankings_parser.add_argument("--source-name", default="hltv-rankings")
    player_stats_parser = subparsers.add_parser("player-stats", help="parse HLTV-like player stat pages into merge-ready CSV")
    player_stats_input = player_stats_parser.add_mutually_exclusive_group(required=True)
    player_stats_input.add_argument("--html", help="path to saved HLTV-like player stats HTML")
    player_stats_input.add_argument("--url", help="HLTV-like player stats URL to fetch through cache")
    player_stats_parser.add_argument("--cache-dir", default=".cs2pickem-cache", help="HTML cache directory for --url")
    player_stats_parser.add_argument("--refresh", action="store_true", help="ignore cached HTML for --url")
    player_stats_parser.add_argument("--date", required=True, help="YYYY-MM-DD snapshot date for rows without per-row dates")
    player_stats_parser.add_argument("--output", required=True, help="player stats CSV output path")
    player_stats_parser.add_argument("--manifest", required=True, help="path for player stats manifest JSON")
    player_stats_parser.add_argument("--source-name", default="hltv-player-stats")
    daily_update_parser = subparsers.add_parser("daily-update", help="run configured daily source updates into a long-lived dataset")
    daily_update_parser.add_argument("--config", required=True, help="JSON config with result source jobs and dataset paths")
    daily_update_parser.add_argument("--output-dir", help="directory for per-job outputs and daily manifest")
    daily_update_parser.add_argument("--refresh", action="store_true", help="ignore cached HTML for URL jobs")
    fivee_parser = subparsers.add_parser("fivee-collect", help="collect public 5E team pages into CSV files")
    fivee_parser.add_argument("--url", action="append", dest="urls", help="5E team page URL, /data/team slug path, or bare team slug; repeatable")
    fivee_parser.add_argument("--url-file", help="text file with one 5E team URL or slug per line")
    fivee_parser.add_argument("--cache-dir", default=".cs2pickem-cache/5e", help="HTML cache directory")
    fivee_parser.add_argument("--output-dir", required=True, help="directory for fivee_teams.csv, fivee_players.csv, fivee_maps.csv, and manifest")
    fivee_parser.add_argument("--refresh", action="store_true", help="ignore cached HTML and refetch public pages")
    fivee_parser.add_argument("--delay-seconds", type=float, default=3.0, help="polite delay between live page fetches")
    fivee_parser.add_argument("--start-date", help="inclusive YYYY-MM-DD lower bound for exported 5E match rows")
    fivee_parser.add_argument("--end-date", help="inclusive YYYY-MM-DD upper bound for exported 5E match rows")
    fivee_results_parser = subparsers.add_parser("fivee-match-results", help="collect public 5E match-result API pages into CSV files")
    fivee_results_parser.add_argument("--cache-dir", default=".cs2pickem-cache/5e-results", help="JSON cache directory")
    fivee_results_parser.add_argument("--output-dir", required=True, help="directory for fivee_match_results.csv, maps, and manifest")
    fivee_results_parser.add_argument("--start-date", required=True, help="inclusive YYYY-MM-DD lower bound")
    fivee_results_parser.add_argument("--end-date", required=True, help="inclusive YYYY-MM-DD upper bound")
    fivee_results_parser.add_argument("--refresh", action="store_true", help="ignore cached JSON and refetch public API pages")
    fivee_results_parser.add_argument("--delay-seconds", type=float, default=0.5, help="polite delay between live API page fetches")
    fivee_results_parser.add_argument("--page-size", type=int, default=100, help="5E result API page size")
    fivee_results_parser.add_argument("--max-pages", type=int, default=1000, help="maximum pages to walk backwards before stopping")
    fivee_results_parser.add_argument("--grades", default="", help="optional 5E event grade filter, empty keeps all grades")
    args = parser.parse_args()

    if args.command == "demo":
        return _emit(run_demo(), None)
    if args.command == "train":
        report = train_evaluate(
            read_matches_csv(args.matches),
            reference_date=args.reference_date,
            top_k=args.top_k,
            cv_folds=args.cv_folds,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            max_age_days=args.max_age_days,
            train_end_date=args.train_end_date,
            validation_end_date=args.validation_end_date,
        )
        return _emit(report, args.output)
    if args.command == "optimize-matches":
        report = optimize_match_predictions(
            read_matches_csv(args.matches),
            reference_date=args.reference_date,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            max_age_days=args.max_age_days,
            top_k_values=_parse_int_list(args.top_k_values),
            epochs_values=_parse_int_list(args.epochs_values),
            candidate_names=_parse_str_list(args.candidates),
            seed=args.seed,
            calibrate=not args.no_calibration,
            rolling_folds=args.rolling_folds,
            market_weight=args.market_weight,
            probability_objective=args.probability_objective,
            elo_modes=_parse_str_list(args.elo_modes),
        )
        return _emit(report, args.output)
    if args.command == "simulate":
        report = simulate_from_team_rows(read_teams_csv(args.teams), simulations=args.simulations, seed=args.seed)
        return _emit(report, args.output)
    if args.command == "pickem":
        report = model_driven_pickems_file(
            history_path=args.history,
            teams_path=args.teams,
            reference_date=args.reference_date,
            profiles_path=args.profiles,
            simulations=args.simulations,
            seed=args.seed,
            top_k=args.top_k,
            epochs=args.epochs,
            stage=args.stage,
            max_age_days=args.max_age_days,
            fixtures_path=args.fixtures,
        )
        return _emit(report, args.output)
    if args.command == "answer-sheet":
        report = build_pickem_answer_sheet_file(
            pickem_report_path=args.pickem_report,
            readiness_report_path=args.readiness_report,
            output_path=None,
            minimum_selection_margin=args.minimum_selection_margin,
        )
        return _emit(report, args.output)
    if args.command == "backtest-pickem":
        report = backtest_pickem_file(
            pickems_path=args.pickems,
            results_path=args.results,
            output_path=None,
            pass_threshold=args.pass_threshold,
        )
        return _emit(report, args.output)
    if args.command == "checkpoint-pickem":
        report = checkpoint_pickem_file(
            pickems_path=args.pickems,
            standings_path=args.standings,
            output_path=None,
        )
        return _emit(report, args.output)
    if args.command == "backtest-forecast":
        report = backtest_forecast_file(
            forecast_report_path=args.forecast_report,
            results_path=args.results,
            output_path=None,
        )
        return _emit(report, args.output)
    if args.command == "backtest-pickem-suite":
        report = backtest_pickem_suite_file(
            suite_path=args.suite,
            output_path=None,
            pass_threshold=args.pass_threshold,
            pass_rate_target=args.pass_rate_target,
        )
        return _emit(report, args.output)
    if args.command == "replay-pickem-suite":
        report = replay_pickem_backtest_suite_file(
            suite_path=args.suite,
            output_path=None,
            pass_threshold=args.pass_threshold,
            pass_rate_target=args.pass_rate_target,
            simulations=args.simulations,
            top_k=args.top_k,
            epochs=args.epochs,
            max_age_days=args.max_age_days,
        )
        return _emit(report, args.output)
    if args.command == "enrich":
        report = enrich_matches_file(args.matches, args.output, args.profiles_output)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "forecast":
        report = forecast_fixtures_file(
            history_path=args.history,
            fixtures_path=args.fixtures,
            reference_date=args.reference_date,
            profiles_path=args.profiles,
            bp_path=args.bp,
            top_k=args.top_k,
            epochs=args.epochs,
            max_age_days=args.max_age_days,
            minimum_margin=args.minimum_margin,
            avoid_player_form_counter_signal=args.avoid_player_form_counter_signal,
            player_form_counter_min_confidence=args.player_form_counter_min_confidence,
        )
        return _emit(report, args.output)
    if args.command == "apply-forecast-policy":
        report = apply_forecast_policy_file(
            forecast_report_path=args.forecast_report,
            fixtures_path=args.fixtures,
            output_path=None,
            minimum_margin=args.minimum_margin,
            avoid_player_form_counter_signal=args.avoid_player_form_counter_signal,
            player_form_counter_min_confidence=args.player_form_counter_min_confidence,
        )
        return _emit(report, args.output)
    if args.command == "merge-odds":
        report = merge_odds_file(args.matches, args.odds, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "merge-players":
        report = merge_player_stats_file(args.matches, args.players, args.output, window_days=args.window_days)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "merge-bp":
        report = merge_bp_file(args.fixtures, args.bp, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "readiness":
        report = audit_readiness_file(
            args.matches,
            args.training_report,
            minimum_rows=args.minimum_rows,
            required_teams=args.required_teams,
            participants_path=args.participants,
            top_teams_path=args.top_teams,
            expected_train_end_date=args.expected_train_end_date,
            expected_validation_end_date=args.expected_validation_end_date,
            minimum_max_age_days=args.minimum_max_age_days,
            sample_reference_date=args.sample_reference_date,
            maximum_sample_age_days=args.maximum_sample_age_days,
            pickem_backtest_report_path=args.pickem_backtest_report,
            pickem_pass_rate_target=args.pickem_pass_rate_target,
            pickem_report_path=args.pickem_report,
            forecast_report_path=args.forecast_report,
            minimum_pickem_simulations=args.minimum_pickem_simulations,
            required_pickem_slots=DEFAULT_PICKEM_SLOTS if args.require_pickem_slots else None,
            minimum_pickem_selection_margin=args.minimum_pickem_selection_margin,
            minimum_pickem_market_adjusted_matchups=args.minimum_pickem_market_adjusted_matchups,
            require_forecast_low_confidence_avoidance=args.require_forecast_low_confidence_avoidance,
            source_manifest_paths=args.source_manifests,
            required_sources=args.required_sources,
            source_reference_time=args.source_reference_time,
            maximum_source_age_hours=args.maximum_source_age_hours,
            require_validation_tuned_weights=args.require_validation_tuned_weights,
        )
        return _emit(report, args.output)
    if args.command == "visualize":
        report = visualize_training_report_file(args.training_report, args.output_dir, prefix=args.prefix)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "pipeline":
        report = run_end_to_end_pipeline(
            history_path=args.history,
            fixtures_path=args.fixtures,
            teams_path=args.teams,
            reference_date=args.reference_date,
            output_dir=args.output_dir,
            odds_path=args.odds,
            players_path=args.players,
            bp_path=args.bp,
            simulations=args.simulations,
            top_k=args.top_k,
            epochs=args.epochs,
            stage=args.stage,
            cv_folds=args.cv_folds,
            window_days=args.window_days,
            max_age_days=args.max_age_days,
            train_end_date=args.train_end_date,
            validation_end_date=args.validation_end_date,
            minimum_rows=args.minimum_rows,
            required_teams=args.required_teams,
            participants_path=args.participants,
            top_teams_path=args.top_teams,
            version_log_path=args.version_log,
            pickem_backtest_report_path=args.pickem_backtest_report,
            pickem_pass_rate_target=args.pickem_pass_rate_target,
            minimum_pickem_simulations=args.minimum_pickem_simulations,
            required_pickem_slots=None if args.skip_pickem_slot_check else DEFAULT_PICKEM_SLOTS,
            minimum_pickem_selection_margin=args.minimum_pickem_selection_margin,
            minimum_pickem_market_adjusted_matchups=args.minimum_pickem_market_adjusted_matchups,
            source_manifest_paths=args.source_manifests,
            required_sources=args.required_sources,
            source_reference_time=args.source_reference_time,
            maximum_source_age_hours=args.maximum_source_age_hours,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "update":
        if args.html:
            report = update_dataset_from_html(
                html_path=args.html,
                output_path=args.output,
                manifest_path=args.manifest,
                version_log_path=args.version_log,
                source_name=args.source_name,
                dataset_path=args.dataset,
                dataset_manifest_path=args.dataset_manifest,
                team_metadata_path=args.team_metadata,
                player_stats_path=args.players,
                player_window_days=args.window_days,
                default_swiss_state=args.default_swiss_state,
            )
        else:
            report = update_dataset_from_url(
                url=args.url,
                cache_dir=args.cache_dir,
                output_path=args.output,
                manifest_path=args.manifest,
                version_log_path=args.version_log,
                source_name=args.source_name,
                refresh=args.refresh,
                dataset_path=args.dataset,
                dataset_manifest_path=args.dataset_manifest,
                team_metadata_path=args.team_metadata,
                player_stats_path=args.players,
                player_window_days=args.window_days,
                default_swiss_state=args.default_swiss_state,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "event-teams":
        if args.html:
            report = update_event_teams_from_html(
                html_path=args.html,
                output_path=args.output,
                manifest_path=args.manifest,
                source_name=args.source_name,
            )
        else:
            report = update_event_teams_from_url(
                url=args.url,
                cache_dir=args.cache_dir,
                output_path=args.output,
                manifest_path=args.manifest,
                source_name=args.source_name,
                refresh=args.refresh,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "rankings":
        if args.html:
            report = update_rankings_from_html(
                html_path=args.html,
                output_path=args.output,
                manifest_path=args.manifest,
                limit=args.limit,
                source_name=args.source_name,
            )
        else:
            report = update_rankings_from_url(
                url=args.url,
                cache_dir=args.cache_dir,
                output_path=args.output,
                manifest_path=args.manifest,
                limit=args.limit,
                source_name=args.source_name,
                refresh=args.refresh,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "player-stats":
        if args.html:
            report = update_player_stats_from_html(
                html_path=args.html,
                output_path=args.output,
                manifest_path=args.manifest,
                default_date=args.date,
                source_name=args.source_name,
            )
        else:
            report = update_player_stats_from_url(
                url=args.url,
                cache_dir=args.cache_dir,
                output_path=args.output,
                manifest_path=args.manifest,
                default_date=args.date,
                source_name=args.source_name,
                refresh=args.refresh,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "daily-update":
        report = daily_update_from_config(
            config_path=args.config,
            output_dir=args.output_dir,
            refresh=args.refresh,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "fivee-collect":
        urls = list(args.urls or [])
        if args.url_file:
            urls.extend(read_urls(args.url_file))
        if not urls:
            raise SystemExit("fivee-collect requires at least one --url or --url-file entry")
        report = collect_fivee_team_pages(
            urls=urls,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            refresh=args.refresh,
            delay_seconds=args.delay_seconds,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "fivee-match-results":
        report = collect_fivee_match_results(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            refresh=args.refresh,
            delay_seconds=args.delay_seconds,
            page_size=args.page_size,
            max_pages=args.max_pages,
            grades=args.grades,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise SystemExit(f"unknown command: {args.command}")


def _emit(payload: object, output_path: str | None) -> int:
    if output_path:
        write_json(output_path, payload)
        return 0
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

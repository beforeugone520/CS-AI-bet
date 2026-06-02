# CS2 Pick'em ML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline Python system for CS2 Major Pick'em prediction: clean match data, generate features, train weighted models, simulate Swiss rounds, and output risk-aware Pick'em picks.

**Architecture:** The project is a dependency-light Python package under `src/cs2pickem` with optional integrations for pandas/sklearn/XGBoost/TensorFlow later. Core behavior is testable with the standard library so the workspace is runnable even before heavy ML packages are installed.

**Tech Stack:** Python 3.9+, `unittest` for current verification, optional extras for `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `requests`, `beautifulsoup4`, and `matplotlib`.

---

## File Structure

- `src/cs2pickem/cleaning.py`: filters invalid/low-value match rows and fills known neutral defaults.
- `src/cs2pickem/features.py`: converts match dictionaries into model-ready numeric features and labels.
- `src/cs2pickem/splitting.py`: provides chronological train/validation/test splits and time-series folds.
- `src/cs2pickem/models.py`: provides baseline logistic, tree-style, boosting-style, neural-style, and weighted ensemble models.
- `src/cs2pickem/swiss.py`: simulates Valve-style seeded Swiss rounds with BO3 advancement/elimination matches.
- `src/cs2pickem/strategy.py`: applies odds adjustment, low-confidence avoidance, upset limits, and Pick'em slot selection.
- `src/cs2pickem/pipeline.py`: coordinates the offline end-to-end demo workflow.
- `src/cs2pickem/cli.py`: exposes `demo`, `train`, and `simulate` entry points.
- `examples/sample_matches.csv`: small local sample for demo runs.
- `tests/test_core.py`: executable specification for the core implementation.

## Tasks

### Task 1: Executable Specification

**Files:**
- Create: `tests/test_core.py`

- [x] **Step 1: Write failing tests**

The tests assert the public behavior for cleaning, feature creation, splitting, model prediction, Swiss simulation, and strategy output.

- [x] **Step 2: Run tests and confirm failure**

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: tests fail because `cs2pickem` modules do not exist yet.

### Task 2: Core Package

**Files:**
- Create: `src/cs2pickem/*.py`
- Create: `pyproject.toml`
- Create: `README.md`

- [x] **Step 1: Implement cleaning and feature modules**

Implement the concrete fields from the project spec: ranking, RMR points, recent win rates, BO1/BO3 mode, map win-rate deltas, player rating/KD/opening/ clutch proxies, H2H, odds, version tag, and Swiss state.

- [x] **Step 2: Implement model and evaluation modules**

Expose four trainable model families and a weighted ensemble with default weights `0.20/0.30/0.35/0.15`.

- [x] **Step 3: Implement Swiss simulation and Pick'em strategy**

Run configurable Monte Carlo simulations, defaulting to 100000 outside tests, and output 3-0, advance, 0-3, and per-team distribution probabilities.

- [x] **Step 4: Run tests and demo**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m cs2pickem.cli demo
```

Expected: tests pass and the demo prints JSON Pick'em output.

## Completion Check

- Data collection fields are represented by schema-compatible CSV/dict inputs and optional dependency declarations.
- Cleaning implements the explicit dirty-data rules from the objective.
- Feature engineering includes static, dynamic, cross, and Swiss-state features.
- Splitting and CV are chronological.
- Four model families and weighted fusion are exposed.
- Swiss Monte Carlo simulation produces 3-0/3-1/3-2/0-3/1-3/2-3, advance, and eliminate probabilities.
- Pick'em strategy applies confidence, odds, upset, and stage rules.

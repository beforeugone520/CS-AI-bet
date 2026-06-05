# Real Swiss Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-data-first Stage 1 Swiss simulator on the GitHub Pages static site.

**Architecture:** Keep all Swiss state transitions in `site/src/swiss.js` and render them through `site/src/render.js`. `site/src/main.js` owns browser state, route loading, and handlers. Static JSON remains the only source of truth; local choices are simulation-only.

**Tech Stack:** Static HTML/CSS, browser ES modules, Node test runner, Python static data export tests.

---

### Task 1: Swiss State Behavior

**Files:**
- Modify: `site/src/swiss.js`
- Modify: `site/tests/swiss.test.mjs`

- [ ] Add tests for replacing a fixture winner, undoing the latest local choice, resetting all local choices, and grouping records by status.
- [ ] Run `node --test site/tests/swiss.test.mjs` and confirm the new tests fail before implementation.
- [ ] Implement selection replacement, `undoSwiss`, `clearSwissSelections`, and `groupSwissRecords`.
- [ ] Run `node --test site/tests/swiss.test.mjs` and confirm all Swiss tests pass.

### Task 2: Stage 1 Workspace

**Files:**
- Modify: `site/src/main.js`
- Modify: `site/src/render.js`
- Modify: `site/styles.css`

- [ ] Wire `onSwissWinner`, `onSwissUndo`, and `onSwissReset` to the new Swiss state.
- [ ] Render selected fixtures, pending fixtures, grouped standings, and Pick'em impact from current local simulation state.
- [ ] Add styles for selected match rows, locked real-data sections, grouped standings, and compact simulator controls.
- [ ] Run `node --test site/tests/*.test.mjs`.

### Task 3: Truthful Fallback Presentation

**Files:**
- Modify: `site/src/render.js`
- Modify: `site/styles.css`

- [ ] Change AI fallback labels to "template fallback" and avoid presenting fallback text as real generated news.
- [ ] Keep Stage 2/3 empty states locked when no real fixtures/bracket data exists.
- [ ] Run `node --test site/tests/*.test.mjs`.

### Task 4: Verify and Deploy

**Files:**
- Generated: `site/data/**`

- [ ] Run `PYTHONPATH=src python3 -m unittest tests.test_site_export tests.test_site_update tests.test_ai_articles -v`.
- [ ] Run `PYTHONPATH=src python3 scripts/export_site_data.py --repo-root . --output-dir site/data`.
- [ ] Run `node --test site/tests/*.test.mjs`.
- [ ] Commit and push `main`; confirm GitHub Pages deployment succeeds.

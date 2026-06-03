# Pick'em Day 1 Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a README-ready SVG that summarizes the IEM Cologne 2026 Stage 1 Pick'em Day 1 checkpoint.

**Architecture:** This is a static documentation feature. The SVG is a self-contained asset in `docs/images/`, and `README.md` embeds it near the current Pick'em answer table so readers can scan prediction status before the poster images.

**Tech Stack:** Markdown, inline SVG, shell XML validation, Git.

---

## File Structure

- Create: `docs/images/pickem-day1-checkpoint.svg`
  - Self-contained dark esports status board with three prediction lanes.
- Modify: `README.md`
  - Insert a short section and image embed after the final fused answer table.
- Modify: `docs/superpowers/plans/2026-06-03-pickem-day1-checkpoint.md`
  - Mark completed implementation steps as work proceeds.

## Task 1: Create The SVG Asset

**Files:**
- Create: `docs/images/pickem-day1-checkpoint.svg`

- [x] **Step 1: Add a self-contained SVG**

Create a 1200x720 SVG with this content:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 720" role="img" aria-labelledby="title desc">
  <title id="title">IEM Cologne 2026 Stage 1 Pick'em Day 1 checkpoint</title>
  <desc id="desc">A dark esports status board showing 3-0 picks, advance picks, and 0-3 picks after two Swiss rounds on 2026-06-02.</desc>
  <!-- Static shapes and text implement the checkpoint board. -->
</svg>
```

The final file must include team blocks for GamerLegion, MIBR, BetBoom, B8, M80, BIG, HEROIC, TYLOO, Gaimin Gladiators, and NRG.

- [x] **Step 2: Validate SVG XML**

Run:

```bash
python3 - <<'PY'
import xml.etree.ElementTree as ET
ET.parse('docs/images/pickem-day1-checkpoint.svg')
print('svg xml ok')
PY
```

Expected output:

```text
svg xml ok
```

## Task 2: Embed The SVG In README

**Files:**
- Modify: `README.md`

- [x] **Step 1: Insert the checkpoint section**

Add this section immediately after the final fused answer table and before `### 预测海报（统一电竞风格）`:

```markdown
### Day 1 Pick'em 状态图

<div align="center">

<img src="docs/images/pickem-day1-checkpoint.svg" width="900" alt="IEM Cologne 2026 Stage 1 Pick'em Day 1 checkpoint" />

</div>
```

- [x] **Step 2: Confirm the README path exists**

Run:

```bash
test -f docs/images/pickem-day1-checkpoint.svg && rg -n "pickem-day1-checkpoint.svg" README.md
```

Expected: command exits successfully and prints the README image line.

## Task 3: Visual And Git Verification

**Files:**
- Inspect: `docs/images/pickem-day1-checkpoint.svg`
- Inspect: `README.md`

- [x] **Step 1: Render or inspect the SVG**

Run one of these commands depending on available tools:

```bash
python3 - <<'PY'
from pathlib import Path
text = Path('docs/images/pickem-day1-checkpoint.svg').read_text()
for required in ['GamerLegion', 'MIBR', 'BetBoom', 'B8', 'M80', 'BIG', 'HEROIC', 'TYLOO', 'Gaimin Gladiators', 'NRG']:
    assert required in text, required
print('svg content ok')
PY
```

Expected output:

```text
svg content ok
```

- [x] **Step 2: Confirm only intended files changed**

Run:

```bash
git status --short
```

Expected changed paths:

```text
 M README.md
 M docs/superpowers/plans/2026-06-03-pickem-day1-checkpoint.md
?? docs/images/pickem-day1-checkpoint.svg
```

- [x] **Step 3: Commit implementation**

Run:

```bash
git add README.md docs/images/pickem-day1-checkpoint.svg docs/superpowers/plans/2026-06-03-pickem-day1-checkpoint.md
git commit -m "docs: add Pick'em Day 1 checkpoint graphic"
```

- [x] **Step 4: Push**

Run:

```bash
git push origin main
```

Expected: push exits successfully and `origin/main` contains the new commits.

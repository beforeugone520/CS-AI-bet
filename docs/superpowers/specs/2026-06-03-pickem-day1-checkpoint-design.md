# Pick'em Day 1 Checkpoint SVG Design

## Purpose

Add a README visual that summarizes how the IEM Cologne 2026 Stage 1 Pick'em selections looked after the first two Swiss rounds on 2026-06-02. The graphic should explain prediction status faster than the existing tables while matching the dark esports tone of the reference bracket screenshot.

## Data Baseline

Use only the Day 1 state already documented in `README.md`:

- `3-0`: GamerLegion at `2-0` remains on track; MIBR at `1-1` is no longer eligible for `3-0`.
- `Advance`: BetBoom, B8, and M80 at `2-0` are strong; BIG at `1-1` remains live; HEROIC and TYLOO at `0-2` are high risk.
- `0-3`: Gaimin Gladiators at `0-2` remains on track; NRG at `1-1` is no longer eligible for `0-3`.

Do not fetch or imply newer match results in this graphic.

## Visual Design

Create `docs/images/pickem-day1-checkpoint.svg` as a self-contained SVG.

Visual thesis: a restrained dark esports Swiss record board with crisp blue rails, compact status blocks, and green/yellow/red semantic highlights.

The SVG should use:

- A dark navy background.
- Three vertical Swiss record columns: `2-0`, `1-1`, and `0-2`.
- A compact KPI strip showing on-track, live, and broken/risk counts.
- Compact team blocks showing team name, Pick'em slot tag, and a short status label.
- Green for on-track or strongly favorable picks.
- Yellow for still-live but risky picks.
- Red for picks whose original slot is already invalid or severely off track.
- Minimal decorative chrome; spacing, contrast, and alignment should carry the layout.

The design must stay readable in GitHub README at desktop widths and should not rely on external fonts, external images, scripts, or team logos.

## README Placement

Insert the SVG in `README.md` after the final fused answer table and before the existing `预测海报（统一电竞风格）` section. This places the checkpoint between the original picks and the poster-style team images.

## Verification

Before commit and push:

- Confirm the SVG parses as XML.
- Confirm the README references the correct relative path.
- Inspect the rendered SVG locally for text overlap and visual hierarchy.
- Confirm the data-driven renderer output matches `docs/images/pickem-day1-checkpoint.svg`.
- Confirm `git status` only includes the intended data, renderer, SVG, test, and documentation edits.
- Commit the changes and push to `origin/main`.

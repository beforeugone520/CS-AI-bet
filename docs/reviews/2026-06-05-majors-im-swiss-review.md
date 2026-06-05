# Majors.im Swiss Selector Review

Date: 2026-06-05

Reference: https://majors.im/2026/cologne

## Screenshot Evidence

- Reference desktop: `docs/reviews/screenshots/majors-desktop-current.png`
- Reference mobile: `docs/reviews/screenshots/majors-mobile-current.png`
- Local desktop after latest changes: `docs/reviews/screenshots/local-desktop-v9.png`
- Local mobile after latest changes: `docs/reviews/screenshots/local-mobile-v10.png`

## Changes Made

- Replaced the dashboard-like shell with a current-event simulator shell for `IEM Cologne Major 2026 Simulator`.
- Removed public navigation to unrelated Overview, AI Desk, and Model Lab pages.
- Changed GitHub Pages automation from scheduled scraping to manual, AI-led workflow dispatch with `ai_update_notes` and optional source refresh.
- Wired `AI_UPDATE_NOTES` into the AI article request payload and generated metadata.
- Converted the Swiss view into a horizontal round board with Stage controls, view switcher, glowing match cards, record ribbons, round-flow arrows, and a bottom Pick'em dock.
- Kept Stage 2 and Stage 3 pages as current-Major placeholders instead of unrelated/fake historical content.
- Tightened the latest pass to match Majors.im proportions more closely: compact topbar on mobile, shorter stage labels, interactive icon-style view controls, 146px desktop match cards, earlier board start position, and a lower Pick'em bar.
- Replaced generated text badges with bundled current-Major PNG team logos from the same event logo set used by Majors.im.

## Browser Verification

- Chromium desktop viewport: 1440 x 768.
- Chromium mobile viewport: 390 x 844.
- Desktop interaction: selecting an unlocked Round 5 winner changes local picks from `0` to `1`.
- View switching: Simple, Minimal, Bracket, and Classic buttons update active state and board layout locally.
- Stage navigation: Stage 2 and Stage 3 hash routes render current-Major placeholder pages.
- Mobile layout: no window-level horizontal scrolling; the Swiss board and Pick'em strip remain horizontally scrollable inside their own containers.
- Logo assets: all bundled team logos and the IEM event icon load with non-zero natural dimensions in Chromium.
- Browser console/page errors: none after adding the static favicon and local assets.

## Remaining Gaps

- Stage/view selector buttons are navigable/visual controls; the alternate view modes are not fully implemented yet.
- The board now uses real team logo assets, but exact Majors.im animations and all alternate view-mode behaviors are not fully cloned.

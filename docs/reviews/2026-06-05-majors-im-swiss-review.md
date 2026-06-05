# Majors.im Swiss Selector Review

Date: 2026-06-05

Reference: https://majors.im/2026/cologne

## Screenshot Evidence

- Reference desktop: `docs/reviews/screenshots/majors-desktop-current.png`
- Reference mobile: `docs/reviews/screenshots/majors-mobile-current.png`
- Local desktop after latest changes: `docs/reviews/screenshots/local-desktop-v8.png`
- Local mobile after latest changes: `docs/reviews/screenshots/local-mobile-v9.png`

## Changes Made

- Replaced the dashboard-like shell with a current-event simulator shell for `IEM Cologne Major 2026 Simulator`.
- Removed public navigation to unrelated Overview, AI Desk, and Model Lab pages.
- Changed GitHub Pages automation from scheduled scraping to manual, AI-led workflow dispatch with `ai_update_notes` and optional source refresh.
- Wired `AI_UPDATE_NOTES` into the AI article request payload and generated metadata.
- Converted the Swiss view into a horizontal round board with Stage controls, view switcher, glowing match cards, record ribbons, round-flow arrows, and a bottom Pick'em dock.
- Kept Stage 2 and Stage 3 pages as current-Major placeholders instead of unrelated/fake historical content.
- Tightened the latest pass to match Majors.im proportions more closely: compact topbar on mobile, shorter stage labels, interactive icon-style view controls, 146px desktop match cards, logo-like team badges, earlier board start position, and a lower Pick'em bar.

## Browser Verification

- Chromium desktop viewport: 1440 x 768.
- Chromium mobile viewport: 390 x 844.
- Desktop interaction: selecting an unlocked Round 5 winner changes local picks from `0` to `1`.
- View switching: Simple, Minimal, Bracket, and Classic buttons update active state and board layout locally.
- Stage navigation: Stage 2 and Stage 3 hash routes render current-Major placeholder pages.
- Mobile layout: no window-level horizontal scrolling; the Swiss board and Pick'em strip remain horizontally scrollable inside their own containers.
- Browser console/page errors: none after adding the static favicon.

## Remaining Gaps

- The local implementation uses logo-like generated badges instead of real team logo image assets.
- Stage/view selector buttons are navigable/visual controls; the alternate view modes are not fully implemented yet.
- The board approximates Majors.im structure and proportions, but exact animations, logo treatment, and all view modes are not cloned.

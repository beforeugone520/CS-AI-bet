# Majors.im Swiss Selector Review

Date: 2026-06-05

Reference: https://majors.im/2026/cologne

## Screenshot Evidence

- Reference desktop: `docs/reviews/screenshots/majors-desktop.png`
- Reference mobile: `docs/reviews/screenshots/majors-mobile.png`
- Local desktop after changes: `docs/reviews/screenshots/local-desktop-v5.png`
- Local mobile after changes: `docs/reviews/screenshots/local-mobile-v5.png`

## Changes Made

- Replaced the dashboard-like shell with a current-event simulator shell for `IEM Cologne Major 2026 Simulator`.
- Removed public navigation to unrelated Overview, AI Desk, and Model Lab pages.
- Changed GitHub Pages automation from scheduled scraping to manual, AI-led workflow dispatch with `ai_update_notes` and optional source refresh.
- Wired `AI_UPDATE_NOTES` into the AI article request payload and generated metadata.
- Converted the Swiss view into a horizontal round board with Stage controls, view switcher, glowing match cards, record ribbons, round-flow arrows, and a bottom Pick'em dock.
- Kept Stage 2 and Stage 3 pages as current-Major placeholders instead of unrelated/fake historical content.

## Browser Verification

- Chromium desktop viewport: 1440 x 768.
- Chromium mobile viewport: 390 x 844.
- Desktop interaction: selecting an unlocked Round 5 winner changes local picks from `0` to `1`.
- Mobile layout: no document-level horizontal overflow; the Swiss board itself remains horizontally scrollable.
- Browser console/page errors: none after adding the static favicon.

## Remaining Gaps

- The local implementation uses initials instead of real team logo assets.
- Stage/view selector buttons are visual controls only; Stage 2 and Stage 3 remain disabled until real bracket data is available.
- The board approximates Majors.im structure and proportions, but exact animations, logo treatment, and all view modes are not cloned.

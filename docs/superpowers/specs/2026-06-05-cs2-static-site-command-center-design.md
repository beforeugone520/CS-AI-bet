# CS2 Static Site Command Center Design

## Goal

Build a GitHub Pages friendly CS2 Major site that combines a majors.im-style stage predictor with an AI esports analysis desk.

The site is a static frontend. It reads generated JSON files from the repository, runs lightweight user interaction in browser JavaScript, and relies on GitHub Actions for scheduled data refresh and AI article generation.

## Product Direction

Use a stage-first hybrid command center.

The home page always opens on the current live stage:

- Stage 1 and Stage 2 show an interactive Swiss predictor.
- Stage 3 shows an interactive playoff bracket predictor.
- A persistent AI Desk sits beside the predictor with generated analysis, Pick'em impact, model notes, and source status.
- Stage tabs allow switching between overview, Stage 1, Stage 2, Stage 3, AI Desk, and Model Lab.

This combines the current-stage immediacy of a matchup simulator with the editorial value of an AI esports news page.

## Visual Direction

Visual thesis:

The product should feel like a live esports intelligence room: deep dark surfaces, high-contrast data, restrained blue state lines, amber critical calls, green locked outcomes, and red broken outcomes.

Design rules:

- Do not make a marketing landing page.
- The first screen is the working surface.
- Use dense but readable information hierarchy.
- Use cards only for interactive items, repeated article entries, and bounded tools.
- Prefer section dividers, status rails, data rows, and bracket boards over decorative card grids.
- Use one primary accent family for data state and one amber accent for critical decisions.
- Use monospace text for records, probabilities, timestamps, model labels, and status tags.
- Use a clear sans-serif for body and analysis text.
- Keep motion lightweight: stage transitions, winner selection, bracket advancement, and AI Desk content swaps.

## Primary Pages

### `/`

Current Stage Command Center.

Content:

- Current event and stage status.
- Last updated timestamp and source status.
- Current-stage predictor.
- AI Desk side rail.
- Next key matches.
- Pick'em status summary.
- Entry points to Stage 1, Stage 2, Stage 3, AI Desk, and Model Lab.

Behavior:

- If current stage format is `swiss`, render the Swiss predictor.
- If current stage format is `playoff`, render the playoff bracket predictor.
- If stage data is not available, render a meaningful future-state panel instead of a blank page.

### `/stage/1`

Stage 1 Swiss page.

Content:

- Swiss standings.
- Round fixtures and completed results.
- Interactive remaining-match simulation from the latest verified standings.
- Pick'em status impact.
- Match detail expansion.
- AI analysis scoped to Stage 1.

### `/stage/2`

Stage 2 Swiss page.

Content:

- Same structure as Stage 1, using Stage 2 teams, standings, fixtures, and results.
- Before Stage 2 starts, show locked/expected participants if known and a waiting state for fixtures.

### `/stage/3`

Stage 3 playoff page.

Content:

- Playoff bracket.
- Quarterfinal, semifinal, final, and champion paths.
- Interactive winner selection.
- Model-generated probabilities when available.
- AI match previews and post-match recaps.

Before bracket release, show:

- Waiting for Stage 2 completion.
- Expected bracket source.
- Last update time.
- Link back to Stage 2 predictor.

### `/ai`

AI Desk archive.

Content:

- Generated articles.
- Filters by stage, match, team, type, and date.
- Article types: round preview, round recap, match preview, model note, Pick'em impact, source status note.
- Each article links back to source data version and relevant match/team/stage.

### `/match/:id`

Match detail page.

Content:

- Teams, current record, map/BO format, source status.
- Model probability if available.
- Market signal if available.
- Player form/status notes if available.
- Pick'em impact.
- AI short analysis.

### `/team/:id`

Team detail page.

Content:

- Current stage status.
- Record and path.
- Completed and upcoming matches.
- Advancement/elimination state.
- Model signals and risk notes.
- AI team-related analysis.

### `/model`

Model Lab.

Content:

- Data source explanation.
- Update schedule.
- Backtest summaries.
- Forecast limitations.
- Pick'em strategy explanations.
- AI generation and fallback status.

## Static Data Contract

The frontend reads static JSON only. It does not call external match sources or AI APIs at runtime.

Directory shape:

```text
site/data/
  latest.json
  events/iem-cologne-2026.json
  stages/stage-1.json
  stages/stage-2.json
  stages/stage-3.json
  matches/*.json
  teams/*.json
  pickem/current.json
  ai/articles.json
  ai/headlines.json
  system/source-status.json
```

`latest.json`:

```json
{
  "event_id": "iem-cologne-2026",
  "current_stage": "stage-1",
  "current_view": "swiss",
  "last_updated": "2026-06-05T18:00:00Z",
  "source_status": "primary_success",
  "ai_status": "generated",
  "data_version": "2026-06-05-r4"
}
```

Swiss stage file:

```json
{
  "stage_id": "stage-1",
  "format": "swiss",
  "status": "live",
  "teams": [],
  "standings": [],
  "rounds": [],
  "fixtures": [],
  "results": [],
  "pickem_impact": {}
}
```

Playoff stage file:

```json
{
  "stage_id": "stage-3",
  "format": "playoff",
  "status": "upcoming",
  "bracket": {
    "quarterfinals": [],
    "semifinals": [],
    "final": []
  },
  "champion_path": {}
}
```

AI articles file:

```json
{
  "generated_at": "2026-06-05T18:03:00Z",
  "model": "configured-via-secret",
  "fallback_used": false,
  "articles": [
    {
      "id": "round-5-watch-big-tyloo",
      "stage": "stage-1",
      "type": "round_preview",
      "title": "BIG 与 TYLOO 是最后两个可补分入口",
      "summary": "Round 5 决胜战将决定 Pick'em 是否还能继续补分。",
      "body": "Round 4 后 Pick'em 状态为 4 locked / 2 alive / 4 broken。BIG 与 TYLOO 仍能把 advance 槽位补成 locked，因此 Round 5 的这两条路径是当前阶段的核心关注点。",
      "source_data_version": "2026-06-05-r4"
    }
  ]
}
```

## Scheduled Update Design

Use GitHub Actions.

Schedule:

- Run daily at 02:00 Beijing time.
- GitHub Actions cron uses UTC, so the cron is `0 18 * * *`.
- Include `workflow_dispatch` for manual refresh.

Workflow:

```text
GitHub Actions
  -> scripts/update_site_data.py
  -> fetch/parse schedules, results, standings
  -> cross-check sources
  -> write site/data/*.json
  -> scripts/generate_ai_articles.py
  -> write AI article JSON
  -> deploy GitHub Pages
```

Source priority:

1. Primary public event sources such as HLTV, Liquipedia, or official event pages.
2. Cross-check sources such as esports.gg or bo3.gg.
3. 5E as fallback.
4. Previously generated valid JSON if all sources fail.

Failure behavior:

- Do not overwrite the last valid site data with empty or failed data.
- Write source status metadata for partial failures.
- Show source status visibly in the frontend.
- If AI generation fails, use deterministic template fallback articles.

## AI Generation Design

AI analysis is generated only in GitHub Actions.

Secrets:

- Store the API key in GitHub Secrets as `AI_API_KEY`.
- Store the base URL and model name as configuration or non-secret workflow env.
- Never commit API keys.
- Never expose API keys in frontend JavaScript.
- Do not log authorization headers.

Generation:

- Read latest standings, fixtures, Pick'em checkpoint, forecast reports, and source status.
- Generate short esports-style articles:
  - current-stage headline
  - next-match preview
  - Pick'em impact
  - model risk note
  - round recap
- Save generated output to `site/data/ai/articles.json` and `site/data/ai/headlines.json`.

Fallback:

- If API call fails, generate template articles from the same data.
- Mark `fallback_used: true`.
- Keep the site usable.

## Browser Interaction Rules

### Stage 1 and Stage 2 Swiss

The first version simulates from the latest verified standings instead of recomputing the whole Swiss event from Round 1.

Rules:

- User selects winners for remaining fixtures.
- Browser updates wins and losses.
- If a team reaches 3 wins, mark as advanced.
- If a team reaches 3 losses, mark as eliminated.
- Update Pick'em state as locked, alive, broken, or pending.
- Let user undo, reset to latest snapshot, or apply current real results.

Controls:

- Select winner.
- Undo last step.
- Reset to real snapshot.
- Apply real result when available.
- Show Pick'em impact.
- Expand match detail.

### Stage 3 Playoff

Rules:

- User selects winners in quarterfinals.
- Winners advance to semifinals.
- Semifinal winners advance to final.
- Final winner becomes champion.
- Reset and undo are supported.
- If model recommendations are available, user can apply them.

Controls:

- Select winner.
- Undo.
- Reset bracket.
- Use model recommendation.
- Share current prediction state.

### AI Desk Reactions

There are two AI Desk content classes:

- Formal articles generated by scheduled GitHub Actions.
- Local instant explanations generated by browser rules after user interactions.

The browser does not call AI during user interactions. It maps known interaction outcomes to concise explanations, for example:

- If BIG beats NRG, BIG's advance pick becomes locked.
- If BIG loses, BIG's advance pick becomes broken.
- If TYLOO wins, the last alive advance path remains viable.
- If TYLOO loses, that advance pick breaks.

## MVP Scope

Build first:

- Static GitHub Pages app shell.
- Current-stage command center home page.
- Stage 1 and Stage 2 Swiss page templates.
- Stage 3 playoff bracket page template.
- Current-snapshot Swiss simulation.
- Bracket simulation for Stage 3.
- AI Desk article list and right rail.
- Data status bar.
- Empty states for future stages.
- Scheduled update workflow design.
- AI generation script design with template fallback.

Defer:

- User accounts.
- Database.
- Real-time minute-by-minute updates.
- Runtime AI chat.
- Full Round 1 Swiss recomputation.
- Backend API service.

## Acceptance Criteria

- The deployed site works on GitHub Pages as a static site.
- No API key exists in committed files or frontend JavaScript.
- GitHub Actions can be configured to run at 02:00 Beijing time.
- Data update failures preserve the last valid JSON.
- AI failures produce template fallback content.
- Stage 2 and Stage 3 pages have complete future-state screens before data exists.
- Stage 1 and Stage 2 Swiss winner selection updates records, advancement, elimination, and Pick'em state in browser.
- Stage 3 bracket winner selection advances teams through the bracket in browser.
- The home page prioritizes the current stage while keeping AI Desk visible.
- Data source, update time, and fallback status are visible to users.

## Implementation Boundaries

- Keep the frontend purely static.
- Keep data generation in scripts and GitHub Actions.
- Reuse existing project outputs where possible:
  - standings CSVs
  - fixtures CSVs
  - Pick'em JSON reports
  - forecast and backtest reports
  - checkpoint SVG/JSON
- Do not refactor model code unless required for data export.
- Keep generated site data separate from source analysis data.

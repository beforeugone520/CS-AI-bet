# Real Swiss Simulator Design

## Goal

Make the static site behave like a real Swiss matchup simulator for the current Stage 1 state, while clearly separating real data from local user simulation and unknown future stages.

## Scope

- Stage 1 uses repository-exported real standings, completed results, and remaining fixtures.
- Completed results are displayed as locked history and are not editable in the browser.
- Remaining fixtures are interactive. Selecting a winner updates local records, status groups, simulated history, and Pick'em status.
- Selecting a different winner for the same fixture replaces the prior local choice instead of stacking another win/loss.
- Undo removes the most recent local fixture choice. Reset clears all local choices.
- Stage 2 and Stage 3 remain locked until real static JSON contains teams, fixtures, or bracket rows.
- AI fallback content is presented as fallback/template status, not as real generated analysis.

## Data Rules

The browser reads static JSON only. It must not invent teams, fixtures, brackets, article claims, or future matchups. Local simulation state is explicitly labeled as simulation and is not persisted back to the repository.

## UI Shape

Stage 1 becomes the primary workspace:

- A compact status bar shows data freshness and source mode.
- A Swiss board shows the remaining fixtures with winner buttons.
- A simulation strip shows selected fixture winners, undo, and reset.
- A standings section groups teams by advanced, live, and eliminated records.
- Pick'em impact updates from the simulated records.

## Testing

Browser logic tests cover replacement selection, undo/reset behavior, and status grouping. Existing Python export tests continue to verify that Stage 2/3 stay empty when real data is unavailable.

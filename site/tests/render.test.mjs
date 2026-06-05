import assert from "node:assert/strict";
import test from "node:test";
import { renderPredictor } from "../src/render.js";

test("renderPredictor renders Swiss rounds as a horizontal round board", () => {
  const root = fakeRoot();
  const stage = {
    stage_id: "stage-1",
    format: "swiss",
    fixtures: [
      {
        team1: "NRG",
        team2: "BIG",
        swiss_round: 5,
        team1_record: "2-2",
        team2_record: "2-2",
        note: "2-2 decider"
      }
    ],
    results: [
      {
        round: "1",
        team1: "M80",
        team2: "Lynn Vision",
        winner: "M80",
        match_score: "1-0"
      }
    ],
    rounds: [
      {
        round: "1",
        results: [
          {
            round: "1",
            team1: "M80",
            team2: "Lynn Vision",
            winner: "M80",
            match_score: "1-0"
          }
        ],
        fixtures: []
      },
      {
        round: "5",
        results: [],
        fixtures: [
          {
            team1: "NRG",
            team2: "BIG",
            swiss_round: 5,
            team1_record: "2-2",
            team2_record: "2-2",
            note: "2-2 decider"
          }
        ]
      }
    ],
    standings: [
      { team: "M80", wins: 3, losses: 1, status: "advanced" },
      { team: "NRG", wins: 2, losses: 2, status: "alive" },
      { team: "BIG", wins: 2, losses: 2, status: "alive" },
      { team: "Lynn Vision", wins: 2, losses: 2, status: "alive" }
    ],
    simulation: {
      history: [],
      selected_by_key: {},
      groups: {
        advanced: [{ team: "M80", wins: 3, losses: 1, status: "advanced" }],
        live: [
          { team: "BIG", wins: 2, losses: 2, status: "alive" },
          { team: "Lynn Vision", wins: 2, losses: 2, status: "alive" },
          { team: "NRG", wins: 2, losses: 2, status: "alive" }
        ],
        eliminated: []
      }
    }
  };

  renderPredictor(root, stage, handlers(), null);

  assert.match(root.innerHTML, /swiss-round-board/);
  assert.match(root.innerHTML, /round-flow-arrow/);
  assert.match(root.innerHTML, /stage-control-row/);
  assert.match(root.innerHTML, /view-switcher/);
  assert.match(root.innerHTML, /pickem-dock/);
  assert.match(root.innerHTML, /team-mark/);
  assert.match(root.innerHTML, /match-score/);
  assert.match(root.innerHTML, /Round 1/);
  assert.match(root.innerHTML, /Round 5/);
  assert.match(root.innerHTML, /locked-match/);
  assert.match(root.innerHTML, /fixture-match/);
  assert.match(root.innerHTML, /data-winner="BIG"/);
  assert.equal((root.innerHTML.match(/locked-match/g) || []).length, 1);
});

test("renderPredictor renders future current-major stage pages with stage links", () => {
  const root = fakeRoot();
  const stage = {
    stage_id: "stage-2",
    format: "swiss",
    status: "upcoming",
    empty_state: {
      title: "等待赛程生成",
      message: "Stage 2 尚未开赛，等待 Stage 1 最终晋级队伍。",
      next_update: "手动 AI 更新时检查真实赛程"
    }
  };

  renderPredictor(root, stage, handlers(), null);

  assert.match(root.innerHTML, /future-stage-page/);
  assert.ok(root.innerHTML.includes('href="#/stage/1"'));
  assert.ok(root.innerHTML.includes('href="#/stage/2"'));
  assert.ok(root.innerHTML.includes('href="#/stage/3"'));
  assert.match(root.innerHTML, /Only IEM Cologne Major 2026 data/);
});

function fakeRoot() {
  return {
    innerHTML: "",
    querySelectorAll() {
      return [];
    },
    querySelector() {
      return null;
    }
  };
}

function handlers() {
  return {
    onSwissWinner() {},
    onSwissUndo() {},
    onSwissReset() {},
    onBracketWinner() {}
  };
}

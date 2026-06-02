import csv
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


FIVEE_TEAM_FIXTURE = """
<html>
  <body>
    <div class="datas-page datas-team-page">
      <p class="bread-crumb"><span class="cur">Case</span></p>
      <div class="team-basic">
        <img class="vm" src="https://static.5eplay.com/images/flag-big/br.gif">
        <p class="name"><span title="Case">Case</span></p>
        <div class="ranks">
          <ul>
            <li><span class="val fs24">151</span><span class="label">世界排名</span></li>
            <li><span class="val fs24">46</span><span class="label">南美洲</span></li>
          </ul>
        </div>
      </div>
      <div class="floatl team-players">
        <ul>
          <li><a href="/data/player/steel"><img alt="steel"><p><span>steel</span></p></a></li>
          <li><a href="/data/player/leo-drk"><img alt="leo_drk"><p><span>leo_drk</span></p></a></li>
        </ul>
      </div>
      <div class="clearfix val-shows">
        <span class="val fs36">42%</span><span class="label">胜率</span>
        <p class="win-data"><span class="win green">75</span>胜 <span class="flat">0</span>平 <span class="lose red">104</span>负</p>
        <table>
          <tr><td><span class="val fs36 purple">0.99</span><span class="label">Rating</span></td></tr>
          <tr><td><span class="val fs36 purple">0.98</span><span class="label">K/D</span></td></tr>
          <tr><td><span class="val fs36 white">179</span><span class="label">地图数</span></td></tr>
        </table>
      </div>
      <p class="map-name">de_inferno</p>
      <p class="map-name">de_mirage</p>
      <table class="fs14 tb-map hide">
        <tr><td colspan="2">胜 / 负 / 平</td><td>21 / 28 / 0</td></tr>
        <tr><td colspan="2">胜率</td><td>43%</td></tr>
        <tr><td colspan="2">总回合数</td><td>1283</td></tr>
        <tr><td colspan="2">取得首杀后回合胜率</td><td>68.1%</td></tr>
        <tr><td colspan="2">被首杀后回合胜率</td><td>26.7%</td></tr>
        <tr><td>最近一场大胜</td><td class="win green">16:3</td><td>Nexus</td></tr>
        <tr><td>最近一场惨败</td><td class="lose red">3:16</td><td>CopenhagenFlames</td></tr>
      </table>
      <table class="fs14 tb-map hide">
        <tr><td colspan="2">胜 / 负 / 平</td><td>8 / 21 / 0</td></tr>
        <tr><td colspan="2">胜率</td><td>28%</td></tr>
      </table>
      <script>
        FiveEplay.datas.teamDetail.init({'session' :[
          {"event_id":"1","event_name":"Recent Cup","event_alias":"recent-cup","session_id":"9001","session_section":"小组赛","format":"0","session_start_time":"1767225600","team1_score":"13","team2_score":"9","is_win":1,"is_match_stats":1,"opponent":{"team_name":"Bravo","team_tag":"Bravo","team_alias":"bravo"}},
          {"event_id":"2","event_name":"Old Cup","event_alias":"old-cup","session_id":"9002","session_section":"淘汰赛","format":"2","session_start_time":"1704067200","team1_score":"0","team2_score":"2","is_win":0,"is_match_stats":0,"opponent":{"team_name":"Delta","team_tag":"Delta","team_alias":"delta"}}
        ]});
      </script>
    </div>
  </body>
</html>
"""


FIVEE_RESULTS_FIXTURE = {
    "success": True,
    "errcode": 0,
    "message": None,
    "data": {
        "matches": [
            {
                "mc_info": {
                    "id": "csgo_mc_2395021",
                    "format": "3",
                    "grade": "5",
                    "plan_ts": "1780219200",
                    "round_name": "决赛",
                    "tt_stage": "淘汰赛",
                    "tt_stage_desc": "淘汰赛 败者组 决赛",
                    "t1_info": {"id": "hltv_team_13479", "disp_name": "Omega", "rank": "42"},
                    "t2_info": {"id": "hltv_team_12194", "disp_name": "Rune Eaters", "rank": "51"},
                },
                "state": {
                    "status": "2",
                    "t1_score": "1",
                    "t2_score": "2",
                    "t1_odds": "1.78",
                    "t2_odds": "1.93",
                    "bout_states": [
                        {"bout_num": "1", "map_name": "Dust2", "result": "t1", "t1_score": "13", "t2_score": "8", "status": "2"},
                        {"bout_num": "2", "map_name": "Ancient", "result": "t2", "t1_score": "7", "t2_score": "13", "status": "2"},
                        {"bout_num": "3", "map_name": "Mirage", "result": "t2", "t1_score": "8", "t2_score": "13", "status": "2"},
                    ],
                },
                "tt_info": {
                    "id": "csgo_tt_9210",
                    "disp_name": "FRAG 第17赛季",
                    "grade": "5",
                    "grade_label": "C级赛事",
                    "status": "past",
                },
            }
        ]
    },
}


class FiveECollectorTests(unittest.TestCase):
    def test_fivee_team_parser_extracts_team_players_and_maps(self):
        from cs2pickem.fivee import FiveETeamParser

        page = FiveETeamParser().parse_team_html(FIVEE_TEAM_FIXTURE, "https://csgo.5eplay.com/data/team/case")

        self.assertEqual(page.team["status"], "ok")
        self.assertEqual(page.team["team"], "Case")
        self.assertEqual(page.team["world_rank"], 151)
        self.assertEqual(page.team["regional_rank"], 46)
        self.assertEqual(page.team["regional_label"], "南美洲")
        self.assertAlmostEqual(page.team["winrate"], 0.42)
        self.assertEqual(page.team["wins"], 75)
        self.assertEqual(page.team["losses"], 104)
        self.assertEqual(page.team["rating"], 0.99)
        self.assertEqual(page.team["kd"], 0.98)
        self.assertEqual(page.team["maps"], 179)
        self.assertEqual([row["player"] for row in page.players], ["steel", "leo_drk"])
        self.assertEqual(page.players[0]["source_player_url"], "https://csgo.5eplay.com/data/player/steel")
        self.assertEqual(len(page.maps), 2)
        self.assertEqual(page.maps[0]["map"], "de_inferno")
        self.assertEqual(page.maps[0]["wins"], 21)
        self.assertEqual(page.maps[0]["losses"], 28)
        self.assertAlmostEqual(page.maps[0]["opening_winrate"], 0.681)
        self.assertEqual(page.maps[0]["last_big_loss_opponent"], "CopenhagenFlames")
        self.assertEqual(len(page.matches), 2)
        self.assertEqual(page.matches[0]["date"], "2026-01-01")
        self.assertEqual(page.matches[0]["opponent"], "Bravo")
        self.assertEqual(page.matches[0]["best_of"], 1)
        self.assertEqual(page.matches[0]["team_score"], 13)
        self.assertEqual(page.matches[0]["opponent_score"], 9)
        self.assertEqual(page.matches[0]["won"], 1)

    def test_fivee_collect_writes_csvs_and_manifest(self):
        from cs2pickem.fivee import collect_fivee_team_pages

        calls = []

        def fetcher(url, headers):
            calls.append(url)
            return FIVEE_TEAM_FIXTURE

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "fivee")
            manifest = collect_fivee_team_pages(
                urls=["case"],
                cache_dir=os.path.join(tmpdir, "cache"),
                output_dir=output_dir,
                refresh=False,
                delay_seconds=0,
                start_date="2025-12-01",
                end_date="2026-06-01",
                fetcher=fetcher,
            )
            with open(manifest["teams_path"], newline="", encoding="utf-8") as handle:
                teams = list(csv.DictReader(handle))
            with open(manifest["players_path"], newline="", encoding="utf-8") as handle:
                players = list(csv.DictReader(handle))
            with open(manifest["maps_path"], newline="", encoding="utf-8") as handle:
                maps = list(csv.DictReader(handle))
            with open(manifest["matches_path"], newline="", encoding="utf-8") as handle:
                matches = list(csv.DictReader(handle))
            with open(manifest["manifest_path"], encoding="utf-8") as handle:
                disk_manifest = json.load(handle)

        self.assertEqual(calls, ["https://csgo.5eplay.com/data/team/case"])
        self.assertEqual(manifest["ok_pages"], 1)
        self.assertEqual(disk_manifest["teams"], 1)
        self.assertEqual(teams[0]["team"], "Case")
        self.assertEqual(players[0]["player"], "steel")
        self.assertEqual(maps[0]["map"], "de_inferno")
        self.assertEqual(matches[0]["event"], "Recent Cup")
        self.assertEqual(len(matches), 1)

    def test_fivee_team_parser_marks_non_team_pages_unparsed(self):
        from cs2pickem.fivee import FiveETeamParser

        page = FiveETeamParser().parse_team_html("<html><title>5EPlay</title></html>", "https://csgo.5eplay.com/data/team/missing")

        self.assertEqual(page.team["status"], "unparsed")
        self.assertEqual(page.players, [])
        self.assertEqual(page.maps, [])

    def test_fivee_result_parser_extracts_matches_and_maps(self):
        from cs2pickem.fivee import parse_fivee_match_results

        matches, maps = parse_fivee_match_results(FIVEE_RESULTS_FIXTURE)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["date"], "2026-05-31")
        self.assertEqual(matches[0]["match_id"], "csgo_mc_2395021")
        self.assertEqual(matches[0]["team1"], "Omega")
        self.assertEqual(matches[0]["team2"], "Rune Eaters")
        self.assertEqual(matches[0]["winner"], "Rune Eaters")
        self.assertEqual(matches[0]["best_of"], 3)
        self.assertEqual(matches[0]["event"], "FRAG 第17赛季")
        self.assertEqual(matches[0]["team1_match_score"], 1)
        self.assertEqual(matches[0]["team2_match_score"], 2)
        self.assertEqual(len(maps), 3)
        self.assertEqual(maps[0]["map"], "Dust2")
        self.assertEqual(maps[0]["winner"], "Omega")
        self.assertEqual(maps[2]["team2_score"], 13)

    def test_fivee_match_result_collector_pages_backwards_and_filters_dates(self):
        from cs2pickem.fivee import collect_fivee_match_results

        calls = []
        old_page = {
            "success": True,
            "data": {
                "matches": [
                    {
                        "mc_info": {
                            "id": "csgo_mc_old",
                            "plan_ts": "1764493200",
                            "t1_info": {"id": "team_old_1", "disp_name": "Old One"},
                            "t2_info": {"id": "team_old_2", "disp_name": "Old Two"},
                        },
                        "state": {"status": "2", "t1_score": "1", "t2_score": "0", "bout_states": []},
                        "tt_info": {"id": "csgo_tt_old", "disp_name": "Old Cup"},
                    }
                ]
            },
        }

        def fetcher(url, headers):
            calls.append(url)
            return json.dumps(FIVEE_RESULTS_FIXTURE if len(calls) == 1 else old_page, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "results")
            manifest = collect_fivee_match_results(
                cache_dir=os.path.join(tmpdir, "cache"),
                output_dir=output_dir,
                start_date="2025-12-01",
                end_date="2026-06-01",
                refresh=False,
                delay_seconds=0,
                page_size=1,
                max_pages=3,
                fetcher=fetcher,
            )
            with open(manifest["matches_path"], newline="", encoding="utf-8") as handle:
                matches = list(csv.DictReader(handle))
            with open(manifest["maps_path"], newline="", encoding="utf-8") as handle:
                maps = list(csv.DictReader(handle))

        self.assertIn("page_token=2026-06-02%2000%3A00%3A00%2C", calls[0])
        self.assertEqual(manifest["pages_fetched"], 2)
        self.assertEqual(manifest["matches"], 1)
        self.assertEqual(matches[0]["match_id"], "csgo_mc_2395021")
        self.assertEqual(len(maps), 3)

    def test_fivee_result_maps_winner_follows_round_score_over_result_field(self):
        """5E 源数据偶尔把 bout.result 标成输家(实测 6 张图如此)。

        地图胜负应以回合分为准(分高者胜)，与 match 层判定逻辑一致，
        而不是盲目信任源数据的 result 字段。
        """
        from cs2pickem.fivee import parse_fivee_match_results

        payload = {
            "success": True,
            "data": {
                "matches": [
                    {
                        "mc_info": {
                            "id": "csgo_mc_bad_result",
                            "format": "1",
                            "plan_ts": "1780219200",
                            "t1_info": {"id": "t_a", "disp_name": "M80"},
                            "t2_info": {"id": "t_b", "disp_name": "NIC"},
                        },
                        "state": {
                            "status": "2",
                            "t1_score": "13",
                            "t2_score": "7",
                            "bout_states": [
                                # result 错标成 t2，但回合分 13:7 明确是 t1 (M80) 赢
                                {"bout_num": "1", "map_name": "Mirage", "result": "t2",
                                 "t1_score": "13", "t2_score": "7", "status": "2"},
                            ],
                        },
                        "tt_info": {"id": "tt", "disp_name": "罗马帝国之声杯"},
                    }
                ]
            },
        }

        matches, maps = parse_fivee_match_results(payload)

        self.assertEqual(matches[0]["winner"], "M80")
        self.assertEqual(len(maps), 1)
        self.assertEqual(maps[0]["winner"], "M80")

    def test_fivee_result_maps_winner_falls_back_to_result_field_when_score_tied(self):
        """回合分缺失或打平(如 forfeit/Default 占位图)时，仍回退到 result 字段。"""
        from cs2pickem.fivee import parse_fivee_match_results

        payload = {
            "success": True,
            "data": {
                "matches": [
                    {
                        "mc_info": {
                            "id": "csgo_mc_forfeit",
                            "format": "1",
                            "plan_ts": "1780219200",
                            "t1_info": {"id": "t_a", "disp_name": "Alpha"},
                            "t2_info": {"id": "t_b", "disp_name": "Beta"},
                        },
                        "state": {
                            "status": "2",
                            "t1_score": "0",
                            "t2_score": "1",
                            "bout_states": [
                                {"bout_num": "1", "map_name": "Default", "result": "t2",
                                 "t1_score": "0", "t2_score": "0", "status": "2"},
                            ],
                        },
                        "tt_info": {"id": "tt", "disp_name": "Forfeit Cup"},
                    }
                ]
            },
        }

        _, maps = parse_fivee_match_results(payload)

        self.assertEqual(maps[0]["winner"], "Beta")


if __name__ == "__main__":
    unittest.main()

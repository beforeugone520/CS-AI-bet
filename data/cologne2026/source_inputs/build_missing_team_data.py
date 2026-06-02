"""Consolidate web-fetched (Liquipedia) recent results for the 4 teams missing from the
5E corpus into a valid-data CSV, and summarize their real recent form."""
import csv

# (date, team, opponent, won?, score, event)  — fetched from Liquipedia /Matches, window 2025-12..2026-06
DATA = {
"GamerLegion": [
 ("2026-01-03","Sharks Esports",True,"2-0","FRAG Miami 2 2026"),
 ("2026-01-03","Beyond Limits",True,"2-0","FRAG Miami 2 2026"),
 ("2026-01-04","Team Venom",True,"2-1","FRAG Miami 2 2026"),
 ("2026-01-04","NRG",False,"0-2","FRAG Miami 2 2026"),
 ("2026-01-04","M80",True,"2-0","FRAG Miami 2 2026"),
 ("2026-05-11","SINNERS Esports",True,"2-1","IEM Atlanta 2026"),
 ("2026-05-12","Natus Vincere",False,"1-2","IEM Atlanta 2026"),
 ("2026-05-13","Team Liquid",True,"2-0","IEM Atlanta 2026"),
 ("2026-05-13","Astralis",True,"2-0","IEM Atlanta 2026"),
 ("2026-05-15","PaiN Gaming",True,"2-1","IEM Atlanta 2026"),
 ("2026-05-16","Legacy",True,"2-1","IEM Atlanta 2026"),
 ("2026-05-17","Natus Vincere",False,"0-3","IEM Atlanta 2026"),
],
"BetBoom": [
 ("2025-12-06","Dynamo Eclot",True,"2-1","Galaxy Battle 2025"),
 ("2025-12-07","Oramond",True,"2-0","Galaxy Battle 2025"),
 ("2025-12-08","FUT Esports",False,"0-2","Galaxy Battle 2025"),
 ("2025-12-12","Johnny Speeds",True,"2-1","CCT EU Series #12"),
 ("2025-12-13","Ex-Betera Esports",False,"0-2","CCT EU Series #12"),
 ("2026-01-16","PaiN Gaming",False,"0-2","BLAST Bounty Winter Qualifier"),
 ("2026-02-11","SINNERS Esports",True,"2-1","IEM Atlanta 2026 Qualifier"),
 ("2026-02-22","Gentle Mates",False,"1-2","Roman Imperium Cup V"),
 ("2026-02-22","Fnatic",True,"2-1","Roman Imperium Cup V"),
 ("2026-03-25","Eternal Fire",True,"2-0","BC.Game Masters Championship"),
 ("2026-03-25","Monte",True,"2-0","BC.Game Masters Championship"),
 ("2026-03-26","SINNERS Esports",False,"0-2","BC.Game Masters Championship"),
 ("2026-03-30","SINNERS Esports",True,"2-1","Roman Imperium Cup VII"),
 ("2026-03-30","G2 Esports",True,"2-1","Roman Imperium Cup VII"),
 ("2026-03-30","BESTIA",True,"2-0","Roman Imperium Cup VII"),
 ("2026-04-01","HEROIC",True,"2-1","Stake Ranked Episode 1"),
 ("2026-04-02","G2 Esports",False,"1-2","Stake Ranked Episode 1"),
 ("2026-04-03","HEROIC",True,"2-1","Stake Ranked Episode 1"),
 ("2026-04-04","GamerLegion",True,"2-0","Stake Ranked Episode 1"),
 ("2026-04-04","G2 Esports",False,"1-2","Stake Ranked Episode 1"),
 ("2026-05-11","B8",True,"2-1","IEM Atlanta 2026"),
 ("2026-05-12","Team Vitality",True,"2-1","IEM Atlanta 2026"),
 ("2026-05-13","PaiN Gaming",True,"2-1","IEM Atlanta 2026"),
 ("2026-05-16","Natus Vincere",False,"0-2","IEM Atlanta 2026"),
 ("2026-05-17","Legacy",False,"0-2","IEM Atlanta 2026"),
],
"HEROIC": [
 ("2026-01-22","FURIA",False,"1-2","BLAST Bounty Winter 2026"),
 ("2026-02-14","FaZe Clan",False,"0-2","PGL Cluj-Napoca 2026"),
 ("2026-02-15","B8",False,"0-2","PGL Cluj-Napoca 2026"),
 ("2026-02-16","3DMAX",True,"2-0","PGL Cluj-Napoca 2026"),
 ("2026-02-17","G2 Esports",False,"0-2","PGL Cluj-Napoca 2026"),
 ("2026-02-27","FlyQuest",True,"2-0","DraculaN Season 5"),
 ("2026-02-28","100 Thieves",True,"2-1","DraculaN Season 5"),
 ("2026-03-01","NRG",True,"2-0","ESL Pro League S23"),
 ("2026-03-02","Astralis",False,"0-2","ESL Pro League S23"),
 ("2026-03-03","Monte",False,"0-2","ESL Pro League S23"),
 ("2026-03-06","MOUZ",False,"1-2","ESL Pro League S23"),
 ("2026-03-07","FUT Esports",False,"1-2","ESL Pro League S23"),
 ("2026-03-08","FURIA",False,"1-2","ESL Pro League S23"),
 ("2026-04-01","BetBoom",False,"1-2","Stake Ranked Episode 1"),
 ("2026-04-03","Ninjas in Pyjamas",True,"2-0","Stake Ranked Episode 1"),
 ("2026-04-03","BetBoom",False,"1-2","Stake Ranked Episode 1"),
 ("2026-04-24","BIG",True,"2-0","CCT Season 3 Global Finals"),
 ("2026-04-25","Monte",True,"2-1","CCT Season 3 Global Finals"),
 ("2026-04-26","Monte",False,"1-3","CCT Season 3 Global Finals"),
 ("2026-05-09","Aurora Gaming",True,"2-0","PGL Astana 2026"),
 ("2026-05-10","FURIA",False,"0-2","PGL Astana 2026"),
 ("2026-05-11","Gentle Mates",False,"0-2","PGL Astana 2026"),
 ("2026-05-12","magic",False,"1-2","PGL Astana 2026"),
],
"Lynn Vision": [
 ("2025-12-18","Team Nemesis Asia",True,"2-0","eXTREMESLAND 2025"),
 ("2025-12-18","Deep Cross Gaming",False,"0-2","eXTREMESLAND 2025"),
 ("2025-12-20","JiJieHao",False,"1-2","eXTREMESLAND 2025"),
 ("2026-01-19","Walk The Talk",True,"2-0","Zhi-Tech Elite Masters Qualifier"),
 ("2026-01-23","Eruption",True,"2-0","Zhi-Tech Elite Masters 2026"),
 ("2026-01-24","JiJieHao",False,"1-2","Zhi-Tech Elite Masters 2026"),
 ("2026-02-06","Alter Ego",True,"2-0","Yuqilin Pinnacle of Battle S2"),
 ("2026-02-07","Morningstar",True,"2-0","Yuqilin Pinnacle of Battle S2"),
 ("2026-02-08","TYLOO",False,"0-2","Yuqilin Pinnacle of Battle S2"),
],
}

COLOGNE = {"GamerLegion","BetBoom","HEROIC","Lynn Vision","M80","SINNERS","FlyQuest","B8","TYLOO","MIBR",
           "THUNDER dOWNUNDER","NRG","Sharks","Sharks Esports","Gaimin Gladiators","BIG","Team Liquid","Liquid"}
TOP = {"Natus Vincere","Team Vitality","G2 Esports","FaZe Clan","MOUZ","FURIA","Astralis","PaiN Gaming",
       "Legacy","Monte","FUT Esports","Aurora Gaming","Gentle Mates"}

rows_out = []
print(f"{'team':<14}{'W-L':>8}{'win%':>7}  notable")
for team, matches in DATA.items():
    w = sum(1 for *_, won, _, _ in [(0,m[1],m[2],m[3],m[4]) for m in matches] if False)  # placeholder
    w = sum(1 for m in matches if m[2])
    l = len(matches) - w
    notable = []
    for date, opp, won, score, event in matches:
        rows_out.append({"date": date, "event": event, "event_tier": "S" if any(k in event for k in ["IEM","PGL","ESL Pro","BLAST","CCT Season 3 Global"]) else "A",
                         "status": "completed", "team1": team, "team2": opp,
                         "winner": team if won else opp, "best_of": 3, "score": score, "source": "liquipedia"})
        if opp in TOP and won:
            notable.append(f"beat {opp}")
    print(f"{team:<14}{str(w)+'-'+str(l):>8}{100*w/len(matches):>6.0f}%  {', '.join(notable[:4]) if notable else '-'}")

OUT = "data/cologne2026/source_inputs/missing_teams_results_2026-06-01.csv"
with open(OUT, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=["date","event","event_tier","status","team1","team2","winner","best_of","score","source"])
    wr.writeheader(); wr.writerows(rows_out)
print(f"\nwrote {len(rows_out)} matches -> {OUT}")

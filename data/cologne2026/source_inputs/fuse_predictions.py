"""Fuse corrected model + 19-expert consensus + market into one high-confidence Pick'em table."""
import json, sys

OUT = "data/cologne2026/predictions/fivee_6m_stage1_2026-06-01"
SRC = "data/cologne2026/source_inputs/expert_predictions_2026-06-01.json"

exp = json.load(open(SRC))
ballots = exp["expert_ballots"]
N = len(ballots)
model = json.load(open(f"{OUT}/pickem_report.json"))["team_probabilities"]
TEAMS = list(model.keys())

# ---- 1. Expert consensus (votes / N) ----
def votes(team, cat):
    return sum(1 for b in ballots.values() if team in b[cat])
E30 = {t: votes(t, "3-0") / N for t in TEAMS}
E03 = {t: votes(t, "0-3") / N for t in TEAMS}
# qualify = picked 3-0 OR advance
EQ  = {t: (votes(t, "3-0") + votes(t, "advance")) / N for t in TEAMS}

# ---- 2. Market: opening-match no-vig win prob -> strength s(t) ----
def am_to_prob(o):  # American odds -> implied prob
    return (-o) / (-o + 100) if o < 0 else 100 / (o + 100)
S = {}
for matchup, odds in exp["opening_match_odds_american"].items():
    (a, oa), (b, ob) = list(odds.items())
    pa, pb = am_to_prob(oa), am_to_prob(ob)
    tot = pa + pb
    S[a], S[b] = pa / tot, pb / tot   # de-vig

# ---- 3. Model ----
M30 = {t: model[t]["3-0"] for t in TEAMS}
M03 = {t: model[t]["0-3"] for t in TEAMS}
MQ  = {t: model[t]["advance"] for t in TEAMS}   # advance = qualify (incl 3-0)

# ---- 4. Fuse. Weights: experts 0.40, market 0.30, model 0.30 ----
WE, WK, WM = 0.40, 0.30, 0.30
def mm(d, keys):  # min-max normalize within candidate set
    vals = [d[k] for k in keys]
    lo, hi = min(vals), max(vals)
    return {k: (d[k] - lo) / (hi - lo) if hi > lo else 0.5 for k in keys}

def fuse(e, k, m, keys):
    en, kn, mn = mm(e, keys), mm(k, keys), mm(m, keys)
    return {t: WE * en[t] + WK * kn[t] + WM * mn[t] for t in keys}

# 3-0 fusion: strength helps; 0-3 fusion: weakness (1-s) helps
S_str = S
S_weak = {t: 1 - S[t] for t in TEAMS}
score30_all = fuse(E30, S_str, M30, TEAMS)
score03_all = fuse(E03, S_weak, M03, TEAMS)

# ---- 5. Select 2 / 6 / 2 (no team in two categories) ----
pick_30 = sorted(TEAMS, key=lambda t: score30_all[t], reverse=True)[:2]
rem = [t for t in TEAMS if t not in pick_30]
score03 = fuse(E03, {t: S_weak[t] for t in rem}, M03, rem)
pick_03 = sorted(rem, key=lambda t: score03[t], reverse=True)[:2]
rem2 = [t for t in rem if t not in pick_03]
scoreQ = fuse(EQ, {t: S_str[t] for t in rem2}, MQ, rem2)
pick_adv = sorted(rem2, key=lambda t: scoreQ[t], reverse=True)[:6]

# ---- 6. Confidence: blended probability + cross-signal agreement ----
def agreement(team, cat):
    """How many of the 3 signals rank this team in the top-k for its category."""
    flags = []
    if cat == "3-0":
        flags = [E30[team] >= 0.20, S[team] >= 0.55, M30[team] >= sorted(M30.values())[-4]]
    elif cat == "0-3":
        flags = [E03[team] >= 0.20, S[team] <= 0.45, M03[team] >= sorted(M03.values())[-4]]
    else:
        flags = [EQ[team] >= 0.50, S[team] >= 0.50, MQ[team] >= 0.55]
    return sum(flags)

def conf(team, cat):
    a = agreement(team, cat)
    if cat == "3-0":  base = (WE*E30[team] + WM*M30[team] + WK*max(0,(S[team]-0.5)))
    elif cat == "0-3": base = E03[team]*0.6 + (1-S[team])*0.4
    else: base = EQ[team]*0.5 + MQ[team]*0.3 + S[team]*0.2
    tier = ["Low", "Low", "Medium", "High"][a]
    return round(base, 3), a, tier

print("="*70)
print("FINAL HIGH-CONFIDENCE PICK'EM (model + 19 experts + market fused)")
print(f"weights: experts {WE}, market {WK}, model {WM}  |  experts N={N}")
print("="*70)
result = {"weights": {"experts": WE, "market": WK, "model": WM}, "n_experts": N, "picks": {}}
for cat, picks in [("3-0", pick_30), ("advance", pick_adv), ("0-3", pick_03)]:
    print(f"\n[{cat}]")
    result["picks"][cat] = []
    for t in picks:
        b, a, tier = conf(t, cat)
        v30, v03, vadv = votes(t,"3-0"), votes(t,"0-3"), votes(t,"advance")
        line = (f"  {t:<18} conf={b:.3f} [{tier}, {a}/3 signals agree] | "
                f"experts 3-0/adv/0-3={v30}/{vadv}/{v03} of {N} | "
                f"market_win={S[t]:.2f} | model 3-0/adv/0-3={M30[t]:.2f}/{MQ[t]:.2f}/{M03[t]:.2f}")
        print(line)
        result["picks"][cat].append({
            "team": t, "confidence": b, "tier": tier, "signals_agree": a,
            "expert_votes": {"3-0": v30, "advance": vadv, "0-3": v03},
            "market_win_prob_r1": round(S[t], 3),
            "model": {"3-0": round(M30[t],3), "advance": round(MQ[t],3), "0-3": round(M03[t],3)},
        })
# ---- 7. Swing detection: runner-up + margin for the last-filled 3-0 and advance slot ----
def slot_rows(cat, picks, score_all, candidate_pool):
    """Return per-slot rows; mark the last slot 'swing' if margin to next candidate is small."""
    ranked = sorted(candidate_pool, key=lambda t: score_all[t], reverse=True)
    rows = []
    for i, t in enumerate(picks):
        rows.append({"slot": f"{cat} #{i+1}", "primary": t, "alt": None, "margin": None, "swing": False})
    # runner-up = first candidate not picked
    chosen = set(picks)
    runner = next((t for t in ranked if t not in chosen), None)
    if runner is not None and picks:
        last = picks[-1]
        margin = round(score_all[last] - score_all[runner], 4)
        rows[-1]["alt"] = runner
        rows[-1]["margin"] = margin
        rows[-1]["swing"] = margin < 0.04
    return rows, runner

rows30, alt30 = slot_rows("3-0", pick_30, score30_all, TEAMS)
# advance pool/scores were computed on rem2; recompute full advance ranking on rem2
adv_rows, altadv = slot_rows("advance", pick_adv, scoreQ, rem2)
# 0-3 rows (locked, but still note runner-up)
rows03, alt03 = slot_rows("0-3", pick_03, {t: score03[t] for t in rem}, rem)

result["swing_slots"] = {}
print("\n" + "="*70)
print("INTEGRATED TABLE WITH SWING SLOTS (primary ⇄ alternative)")
print("="*70)
def sig(t):
    return f"exp {votes(t,'3-0')}/{votes(t,'advance')}/{votes(t,'0-3')} | mkt {S[t]:.2f} | mdl {M30[t]:.2f}/{MQ[t]:.2f}/{M03[t]:.2f}"
for cat, rows in [("3-0", rows30), ("advance", adv_rows), ("0-3", rows03)]:
    for r in rows:
        tag = " ⇄ SWING" if r["swing"] else ""
        alt = f"   alt: {r['alt']} ({sig(r['alt'])}) margin={r['margin']}" if r["swing"] else ""
        print(f"{r['slot']:<12} {r['primary']:<18} | {sig(r['primary'])}{tag}")
        if alt: print(alt)
        if r["swing"]:
            result["swing_slots"][r["slot"]] = {"primary": r["primary"], "alternative": r["alt"], "fused_margin": r["margin"]}

# integrated CSV artifact
import csv
with open(f"{OUT}/final_fused_pickem_table_2026-06-01.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["slot","category","pick","alternative_if_swing","fused_margin","confidence","tier","signals_agree",
                "expert_3-0","expert_advance","expert_0-3","market_win_r1","model_3-0","model_advance","model_0-3"])
    for cat, rows in [("3-0", rows30), ("advance", adv_rows), ("0-3", rows03)]:
        for r in rows:
            t = r["primary"]; b,a,tier = conf(t,cat)
            w.writerow([r["slot"],cat,t, r["alt"] if r["swing"] else "", r["margin"] if r["swing"] else "",
                        b,tier,a, votes(t,"3-0"),votes(t,"advance"),votes(t,"0-3"),
                        round(S[t],3),round(M30[t],3),round(MQ[t],3),round(M03[t],3)])

json.dump(result, open(f"{OUT}/final_fused_pickem_2026-06-01.json", "w"), indent=2, ensure_ascii=False)
print(f"\nwritten: {OUT}/final_fused_pickem_2026-06-01.json")
print(f"written: {OUT}/final_fused_pickem_table_2026-06-01.csv")

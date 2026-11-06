"""Agrège les résultats DiffLight en un seul CSV : experiments_log/summary_all_results.csv"""
import glob
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MASK_PCT = {"kmrand0_2": 12.5, "kmrand0_4": 25, "kmrand0_8": 50}

runs = pd.read_csv(f"{ROOT}/experiments_log/difflight_hangzhou12_results.csv").set_index("memo")

rows = []
for memo in sorted(d for d in os.listdir(f"{ROOT}/records") if d.startswith("DiffLight_")):
    row = {"memo": memo}
    if memo in runs.index:
        r = runs.loc[memo]
        row.update(mask=r["pattern"], mask_pct=MASK_PCT[r["pattern"]], seed=r["seed"],
                   status=r["status"], train_duration_h=round(r["duration_s"] / 3600, 2))
    elif memo.startswith("DiffLight_hangzhou12"):
        # run pas encore dans le CSV de pipeline : masque et seed lus dans l'en-tête du log
        with open(f"{ROOT}/experiments_log/{memo}.log") as f:
            head = f.readline()
        row.update(mask=re.search(r"mask=(\S+)", head).group(1), mask_pct="",
                   seed=int(re.search(r"train_seed=(\d+)", head).group(1)), status="RUNNING")
    else:
        row.update(mask="km_wn", mask_pct=25, seed="", status="SUCCESS")
    sdir = glob.glob(f"{ROOT}/summary/{memo}/*/*/*/")
    if sdir:
        d = sdir[0]
        fin = pd.read_csv(d + "final_five_rounds_test_results.csv").iloc[0]
        t = pd.read_csv(d + "test_results.csv")
        row.update(
            n_test_rounds=len(t),
            final5_duration_avg=fin["final_duration_avg"],
            final5_duration_std=fin["final_duration_std"],
            final5_through_avg=fin["final_through_avg"],
            vehicle_in=int(t["vehicle_in"].iloc[-1]),
            best_round=int(t["duration"].idxmin()),
            best_round_duration=round(t["duration"].min(), 2),
            last_round_duration=round(t["duration"].iloc[-1], 2),
            round0_duration=round(t["duration"].iloc[0], 2),
        )
    rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values(["mask_pct", "seed"], key=lambda s: pd.to_numeric(s, errors="coerce"),
                    na_position="last", kind="stable")

# moyenne / écart-type inter-seeds par masque
agg = []
for pct, g in df[(df["seed"] != "") & (df["mask_pct"] != "")].groupby("mask_pct"):
    g = g.dropna(subset=["final5_duration_avg"])
    if g.empty:
        continue
    for label, stat in (("MEAN", g.mean), ("STD", lambda **k: g.std(ddof=0, **k))):
        vals = stat(numeric_only=True)
        a = {"memo": f"{label}_mask{pct}", "mask": g["mask"].iloc[0], "mask_pct": pct,
             "seed": label, "status": f"n={len(g)}"}
        for c in ("train_duration_h", "final5_duration_avg", "final5_through_avg",
                  "best_round_duration", "last_round_duration"):
            a[c] = round(float(vals[c]), 2)
        agg.append(a)

df = pd.concat([df, pd.DataFrame(agg)], ignore_index=True)
out = f"{ROOT}/experiments_log/summary_all_results.csv"
df.to_csv(out, index=False)
print(df.drop(columns="memo").to_string())
print(f"\n-> {out}")

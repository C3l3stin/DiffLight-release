"""
Pipeline de nuit DiffLight — lance plein d'entraînements run_difflight.py.

Conçu pour reproduire les expériences du papier DiffLight sur le Kriging Missing (KM).
- Hangzhou 1 & 2 : 6.25%, 12.50%, 18.75%, 25.00%
- Jinan 1, 2 & 3 : 8.33%, 16.67%, 25.00%

Usage :
    python run_difflight_night.py
    nohup python run_difflight_night.py > nuit_difflight.log 2>&1 &
"""

import os
import time
import subprocess
from datetime import datetime

# ── Configurations ────────────────────────────────────────────────────────────

EXPERIMENTS = []

# 1. Configurations pour Hangzhou (4x4)
hangzhou_datasets = ["hangzhou_1"]
hangzhou_kriging_rates = ["12", "25", "50", "75"]


for rate in hangzhou_kriging_rates:
    EXPERIMENTS.append({
        "name": f"Hangzhou - KM {float(rate):.2f}%",
        "memo": f"DiffLight_Hangzhou_KM_{rate}",
        "dataset_flag": "-hangzhou_1",
        "missing_pattern": f"km_{rate}"
    })

# # 2. Configurations pour Jinan (3x4)
# jinan_datasets = ["jinan_1", "jinan_2", "jinan_3"]
# jinan_kriging_rates = ["0.0833", "0.1667", "0.2500"]

# for ds in jinan_datasets:
#     for rate in jinan_kriging_rates:
#         EXPERIMENTS.append({
#             "name": f"{ds.capitalize()} - KM {float(rate)*100:.2f}%",
#             "memo": f"DiffLight_{ds}_KM_{rate}",
#             "dataset_flag": f"-{ds}",
#             "missing_pattern": f"KM_{rate}"
#         })

# ── Paramètres communs à tous les runs ────────────────────────────────────────
COMMON_ARGS = {
    "-mod": "DiffLight",
    "-gen": "1",
    "-workers": "3",  # Si tu as assez de cœurs, sinon descends à 1 ou 2
}

os.makedirs("experiments_log", exist_ok=True)
LOG_FILE = f"experiments_log/difflight_km_{datetime.now().strftime('%m%d_%H%M')}.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def build_train_args(exp):
    """Traduit une expérience en arguments de run_difflight.py."""
    # On ajoute le dataset flag (ex: -hangzhou_1) qui agit comme un store_true
    a = [
        "-memo", exp["memo"],
        "-missing_pattern", exp["missing_pattern"],
        exp["dataset_flag"]
    ]
    return a


def run_one(exp):
    cmd = ["python", "run_difflight.py"]
    
    # Ajout des arguments communs
    for k, v in COMMON_ARGS.items():
        cmd += [k, str(v)]
        
    # Ajout des arguments spécifiques à l'expérience
    cmd += build_train_args(exp)
    
    log(f"Commande : {' '.join(cmd)}")
    start = time.time()
    
    # Exécution du script
    result = subprocess.run(cmd)
    
    elapsed = time.time() - start
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    ok = result.returncode == 0
    log(f"{'✓ SUCCÈS' if ok else f'✗ ÉCHEC (code {result.returncode})'} — durée {h:02d}h {m:02d}m {s:02d}s")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    total = len(EXPERIMENTS)
    log("=" * 60)
    log(f"PIPELINE DIFFLIGHT (Kriging Missing) — {total} run(s) planifié(s)")
    log(f"Log : {LOG_FILE}")
    log("=" * 60)

    results = []
    overall_start = time.time()
    
    for idx, exp in enumerate(EXPERIMENTS, 1):
        log(f"\n{'─' * 60}")
        log(f"Run {idx}/{total} : {exp['name']}  (memo={exp['memo']})")
        log("─" * 60)
        
        ok = run_one(exp)
        results.append({"name": exp["name"], "success": ok})

    elapsed = time.time() - overall_start
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    
    log(f"\n{'=' * 60}")
    log(f"RÉSUMÉ — durée totale {h:02d}h {m:02d}m {s:02d}s")
    for r in results:
        log(f"  {'✓' if r['success'] else '✗'}  {r['name']}")
    
    n_ok = sum(r["success"] for r in results)
    log(f"  -> {n_ok}/{total} runs réussis")
    log("=" * 60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""15 entraînements DiffLight séquentiels sur Hangzhou_1.

5 masques fixes x 3 graines d'entraînement (0, 1, 2).
Les fichiers de masque ne sont jamais régénérés ni modifiés par ce lanceur.

Préparation des masques aléatoires (une seule fois, seed placement = 0) :
    python make_intersection_mask_50.py --layout random --seed 0 --count 2
    python make_intersection_mask_50.py --layout random --seed 0 --count 4
    python make_intersection_mask_50.py --layout random --seed 0 --count 8
    python make_intersection_mask_50.py --layout center
    python make_intersection_mask_50.py --layout border

Attention : la version fournie de make_intersection_mask_50.py refuse à juste
titre de remplacer une clé déjà présente (pas de --overwrite ici).

Depuis la racine du dépôt DiffLight :
    python run_difflight_sequential.py --dry-run
    nohup python -u run_difflight_sequential.py > experiments_log/pipeline_difflight.log 2>&1 &

run_difflight.py doit avoir num_rounds=15 pour Hangzhou_1.
Chaque run utilise 10 epochs x 1000 mises à jour/epoch = 150k updates.
"""

import argparse
import csv
import datetime as dt
import json
import os
import pickle
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAFFIC_STEM = "anon_4_4_hangzhou_real"
MASK_ROOT = Path("memory/fourphase")
MASKS = (
    ("intersection_12_5", "kmrand0_2", 2),
    ("intersection_25", "kmrand0_4", 4),
    ("intersection_50", "kmrand0_8", 8),
    ("center_4", "kmcenter_4", 4),
    ("border_12", "kmborder_12", 12),
)

# Le script enfant exécute cette amorce dans son propre interpréteur :
# PYTHONHASHSEED seul ne fixe pas les graines de PyTorch/NumPy.
SEED_BOOTSTRAP = """
import os, random, runpy, sys
import numpy as np
import torch
seed = int(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print(f'[seed] training={seed} CUDA={torch.cuda.is_available()}', flush=True)
sys.argv = ['run_difflight.py'] + sys.argv[2:]
runpy.run_path('run_difflight.py', run_name='__main__')
"""


def mask_file(pattern):
    family, _count = pattern.rsplit("_", 1)
    return ROOT / MASK_ROOT / family / f"{TRAFFIC_STEM}_{family}.pkl"


def validate_masks():
    """Vérifie clés, formes, taux et stabilité de chaque fichier de masque."""
    errors = []
    reference_keys = None
    for label, pattern, expected_count in MASKS:
        path = mask_file(pattern)
        if not path.is_file():
            errors.append(f"{pattern}: fichier introuvable : {path}")
            continue
        try:
            # Les .pkl de ce projet sont des données de confiance locales.
            # Ne jamais charger un pickle provenant d'une source inconnue.
            with path.open("rb") as handle:
                masks = pickle.load(handle)
            if not isinstance(masks, dict) or not masks:
                raise ValueError("dictionnaire de trajectoires vide/invalide")
            keys = set(masks)
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise ValueError("clés de trajectoires différentes des autres masques")

            placement = None
            for trajectory, variants in masks.items():
                if pattern not in variants:
                    raise ValueError(f"clé {pattern!r} absente de {trajectory!r}")
                import numpy as np
                array = np.asarray(variants[pattern])
                if array.ndim != 3 or array.shape[0] != 16:
                    raise ValueError(f"forme invalide {trajectory}: {array.shape}")
                if not np.isin(array, [0, 1]).all():
                    raise ValueError(f"valeurs non binaires dans {trajectory}")
                missing = tuple(np.flatnonzero((array == 0).all(axis=(1, 2))).tolist())
                if len(missing) != expected_count:
                    raise ValueError(f"{trajectory}: {len(missing)} != {expected_count} intersections")
                if not np.all(array[list(set(range(16)) - set(missing))] == 1):
                    raise ValueError(f"{trajectory}: masque partiel inattendu")
                if placement is None:
                    placement = missing
                elif missing != placement:
                    raise ValueError(f"placement change selon la trajectoire: {trajectory}")
            print(f"[masque OK] {label}: {pattern}, indices={list(placement)}, {len(keys)} trajectoires", flush=True)
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{pattern}: {exc}")
    if errors:
        raise RuntimeError("Prévalidation des masques échouée :\n - " + "\n - ".join(errors))


def check_training_budget():
    script = ROOT / "run_difflight.py"
    if not script.is_file():
        raise RuntimeError(f"Script introuvable : {script}")
    source = script.read_text(encoding="utf-8")
    match = re.search(r"if in_args\.hangzhou_1:\s*.*?num_rounds\s*=\s*(\d+)", source, re.S)
    if not match or int(match.group(1)) != 15:
        raise RuntimeError("Vérifie num_rounds=15 dans le bloc Hangzhou_1 de run_difflight.py")


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="afficher les 15 commandes sans démarrer")
    parser.add_argument("--resume", action="store_true", help="reprendre les succès du même fichier CSV")
    parser.add_argument("--tag", default="sequential", help="nom de campagne stable pour --resume")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--skip-mask-check", action="store_true", help="pour un dry-run sans masques locaux")
    args = parser.parse_args()
    if args.epochs < 1 or args.steps_per_epoch < 1:
        parser.error("epochs et steps-per-epoch doivent être positifs")
    os.chdir(ROOT)
    check_training_budget()
    if not args.skip_mask_check:
        validate_masks()

    logdir = ROOT / "experiments_log"
    if not args.dry_run:
        logdir.mkdir(exist_ok=True)
    csv_path = logdir / f"difflight_{args.tag}_results.csv"
    successful = set()
    if args.resume and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["status"] == "SUCCESS":
                    successful.add((row["pattern"], int(row["seed"])))

    jobs = [(label, pattern, seed) for label, pattern, _ in MASKS for seed in (0, 1, 2)]
    print(f"{now()} | {len(jobs)} runs séquentiels, seeds RL 0,1,2; placement fixe 0", flush=True)
    for position, (label, pattern, seed) in enumerate(jobs, 1):
        memo = f"DiffLight_{args.tag}_{label}_seed{seed}"
        log = logdir / f"{memo}.log"
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        env["DIFFLIGHT_TRAIN_SEED"] = str(seed)
        cmd = [sys.executable, "-u", "-c", SEED_BOOTSTRAP, str(seed),
               "-hangzhou_1", "-missing_pattern", pattern, "-memo", memo,
               "-epochs", str(args.epochs), "-n_steps_per_epoch", str(args.steps_per_epoch)]
        if (pattern, seed) in successful:
            print(f"[{position}/{len(jobs)}] SKIP déjà terminé : {memo}", flush=True)
            continue
        print(f"[{position}/{len(jobs)}] {memo} | masque={pattern} | seed RL={seed}", flush=True)
        if args.dry_run:
            print("  " + shlex.join([sys.executable, "-u", "-c", "<seed-bootstrap>", str(seed)] + cmd[5:]), flush=True)
            continue
        start = time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            stream.write(f"{now()} | memo={memo} | mask={pattern} | train_seed={seed}\n")
            stream.flush()
            try:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                status = "SUCCESS" if result.returncode == 0 else "FAILED"
                code = result.returncode
            except OSError as exc:
                status, code = "FAILED", -1
                stream.write(f"Lancement impossible: {exc}\n")
        duration = round(time.monotonic() - start, 1)
        print(f"  {status} | {duration / 3600:.2f} h | log={log}", flush=True)
        is_new = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("time", "pattern", "seed", "memo", "status", "returncode", "duration_s", "log"))
            if is_new:
                writer.writeheader()
            writer.writerow({"time": now(), "pattern": pattern, "seed": seed, "memo": memo,
                             "status": status, "returncode": code, "duration_s": duration, "log": str(log)})
        if status != "SUCCESS":
            print("Échec : campagne arrêtée. Corriger puis relancer avec --resume.", flush=True)
            return 1
    print(f"Terminé. Résultats : {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

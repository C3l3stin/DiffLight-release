#!/usr/bin/env python3
"""30 entraînements DiffLight séquentiels : Hangzhou_1 puis Hangzhou_2.

2 datasets x 5 masques fixes x 3 graines d'entraînement (0, 1, 2).
Sur Hangzhou_2, crée les masques manquants à partir d'un masque de référence
déjà présent dans memory/fourphase/hangzhou_2/, puis lie les fichiers depuis
memory/fourphase/ sans recopier le dataset de 4 Go.

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

run_difflight.py doit avoir num_rounds=15 pour les deux datasets.
Chaque run utilise 10 epochs x 1000 mises à jour/epoch = 150k updates.
"""

import argparse
import csv
import datetime as dt
import os
import pickle
import random
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAFFIC_STEM = "anon_4_4_hangzhou_real"
MASK_ROOT = Path("memory/fourphase")
DATASETS = {
    "hangzhou_1": TRAFFIC_STEM,
    "hangzhou_2": f"{TRAFFIC_STEM}_5816",
}
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


def mask_file(pattern, dataset):
    family, _count = pattern.rsplit("_", 1)
    return ROOT / MASK_ROOT / family / f"{DATASETS[dataset]}_{family}.pkl"


def validate_masks(dataset):
    """Vérifie clés, formes, taux et stabilité de chaque fichier de masque."""
    errors = []
    reference_keys = None
    for label, pattern, expected_count in MASKS:
        path = mask_file(pattern, dataset)
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
            print(f"[masque OK] {dataset}/{label}: {pattern}, indices={list(placement)}, {len(keys)} trajectoires", flush=True)
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{pattern}: {exc}")
    if errors:
        raise RuntimeError(f"Prévalidation des masques {dataset} échouée :\n - " + "\n - ".join(errors))


def missing_intersections(layout, count):
    """Même placement seed=0 que make_intersection_mask_50.py."""
    if layout == "kmcenter":
        return [5, 6, 9, 10]
    if layout == "kmborder":
        return [i for i in range(16) if i not in (5, 6, 9, 10)]
    names = [f"intersection_{col}_{row}" for col in range(1, 5) for row in range(1, 5)]
    random.Random(0).shuffle(names)
    selected = names[::-1][:count]
    return [(int(name.split("_")[1]) - 1) * 4 + int(name.split("_")[2]) - 1 for name in selected]


def link_if_absent(source, destination):
    """Ne remplace jamais un fichier ou un lien existant."""
    if destination.exists() or destination.is_symlink():
        if not destination.is_file():
            raise RuntimeError(f"Chemin existant mais invalide : {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())
    print(f"[lien] {destination} -> {source}", flush=True)


def prepare_hangzhou_2():
    """Prépare les masques HZ2 sans toucher à la collecte offline initiale."""
    import numpy as np

    nested = ROOT / MASK_ROOT / "hangzhou_2"
    stem = DATASETS["hangzhou_2"]
    source_traffic = nested / f"{stem}.pkl"
    if not source_traffic.is_file():
        raise RuntimeError(f"Trajectoires Hangzhou 2 introuvables : {source_traffic}")

    # Prend uniquement la structure des trajectoires (clés et dimensions),
    # sans importer les placements masqués de cette référence.
    ref_candidates = []
    for family in ("km_won", "km_wn", "rm"):
        ref_candidates.extend(sorted((nested / family).glob(f"{stem}_{family}.pkl")))
    if not ref_candidates:
        raise RuntimeError(f"Aucun masque de référence Hangzhou 2 dans {nested}/km_won, km_wn ou rm")
    with ref_candidates[0].open("rb") as handle:
        reference = pickle.load(handle)
    if not isinstance(reference, dict) or not reference:
        raise RuntimeError(f"Masque de référence invalide : {ref_candidates[0]}")

    # Les trois taux aléatoires cohabitent dans le même fichier kmrand0.
    generated = {}
    for _label, pattern, count in MASKS:
        path = mask_file(pattern, "hangzhou_2")
        family = pattern.rsplit("_", 1)[0]
        if path.is_file():
            # On garde les éventuels masques existants ; la validation suit.
            continue
        generated.setdefault(path, []).append((pattern, family, count))

    for destination, variants in generated.items():
        output = {}
        for trajectory, ref_variants in reference.items():
            shape = np.asarray(next(iter(ref_variants.values()))).shape
            if len(shape) != 3 or shape[0] != 16:
                raise RuntimeError(f"Forme de masque inattendue : {trajectory} -> {shape}")
            output[trajectory] = {}
            for pattern, family, count in variants:
                mask = np.ones(shape, dtype=np.uint8)
                mask[missing_intersections(family, count), :, :] = 0
                output[trajectory][pattern] = mask
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Fichier temporaire puis renommage, uniquement si destination absente.
        temp = destination.with_name(destination.name + ".building")
        if temp.exists():
            raise RuntimeError(f"Fichier temporaire existant, vérification manuelle : {temp}")
        with temp.open("xb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"Fichier de destination apparu pendant la préparation : {destination} (temporaire : {temp})")
        temp.rename(destination)
        print(f"[créé] {destination} ({', '.join(v[0] for v in variants)})", flush=True)

    # Le run_difflight.py original attend ces fichiers directement sous fourphase.
    link_if_absent(source_traffic, ROOT / MASK_ROOT / f"{stem}.pkl")


def check_training_budget():
    script = ROOT / "run_difflight.py"
    if not script.is_file():
        raise RuntimeError(f"Script introuvable : {script}")
    source = script.read_text(encoding="utf-8")
    for dataset in DATASETS:
        match = re.search(rf"(?:if|elif) in_args\.{dataset}:\s*.*?num_rounds\s*=\s*(\d+)", source, re.S)
        if not match or int(match.group(1)) != 15:
            raise RuntimeError(f"Vérifie num_rounds=15 dans le bloc {dataset} de run_difflight.py")


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="afficher les 30 commandes sans démarrer")
    parser.add_argument("--resume", action="store_true", help="reprendre les succès du même fichier CSV")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS),
                        help="datasets à exécuter, par défaut Hangzhou_1 puis Hangzhou_2")
    parser.add_argument("--tag", default="hangzhou12", help="nom de campagne stable pour --resume")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--skip-mask-check", action="store_true", help="pour un dry-run sans masques locaux")
    args = parser.parse_args()
    if args.epochs < 1 or args.steps_per_epoch < 1:
        parser.error("epochs et steps-per-epoch doivent être positifs")
    os.chdir(ROOT)
    check_training_budget()
    if not args.skip_mask_check:
        validate_masks("hangzhou_1") if "hangzhou_1" in args.datasets else None
        if "hangzhou_2" in args.datasets and not args.dry_run:
            prepare_hangzhou_2()
            validate_masks("hangzhou_2")
        elif "hangzhou_2" in args.datasets:
            print("[dry-run] Hangzhou_2 : préparation/validation des masques différée jusqu'au lancement.", flush=True)

    logdir = ROOT / "experiments_log"
    if not args.dry_run:
        logdir.mkdir(exist_ok=True)
    csv_path = logdir / f"difflight_{args.tag}_results.csv"
    successful = set()
    if args.resume and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["status"] == "SUCCESS":
                    successful.add((row.get("dataset", "hangzhou_1"), row["pattern"], int(row["seed"])))

    jobs = [(dataset, label, pattern, seed) for dataset in args.datasets
            for label, pattern, _ in MASKS for seed in (0, 1, 2)]
    print(f"{now()} | {len(jobs)} runs séquentiels, seeds RL 0,1,2; placement fixe 0", flush=True)
    for position, (dataset, label, pattern, seed) in enumerate(jobs, 1):
        memo = f"DiffLight_{args.tag}_{dataset}_{label}_seed{seed}"
        log = logdir / f"{memo}.log"
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        env["DIFFLIGHT_TRAIN_SEED"] = str(seed)
        cmd = [sys.executable, "-u", "-c", SEED_BOOTSTRAP, str(seed),
               f"-{dataset}", "-missing_pattern", pattern, "-memo", memo,
               "-epochs", str(args.epochs), "-n_steps_per_epoch", str(args.steps_per_epoch)]
        if (dataset, pattern, seed) in successful:
            print(f"[{position}/{len(jobs)}] SKIP déjà terminé : {memo}", flush=True)
            continue
        print(f"[{position}/{len(jobs)}] {memo} | dataset={dataset} | masque={pattern} | seed RL={seed}", flush=True)
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
            writer = csv.DictWriter(file, fieldnames=("time", "dataset", "pattern", "seed", "memo", "status", "returncode", "duration_s", "log"))
            if is_new:
                writer.writeheader()
            writer.writerow({"time": now(), "dataset": dataset, "pattern": pattern, "seed": seed, "memo": memo,
                             "status": status, "returncode": code, "duration_s": duration, "log": str(log)})
        if status != "SUCCESS":
            print("Échec : campagne arrêtée. Corriger puis relancer avec --resume.", flush=True)
            return 1
    print(f"Terminé. Résultats : {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

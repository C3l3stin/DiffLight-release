"""
seed_eval_colight.py — seed-eval pour CoLight (TensorFlow/Keras).

Pour CHAQUE config sous --records_dir : cherche le round au MEILLEUR avg_reward,
charge les poids correspondants (model/<config>/<run>/round_<best>_inter_0.h5),
et évalue sur plusieurs seeds CityFlow (et éventuellement un autre flux).

⚠ CoLight ≠ baselines PyTorch : les .h5 sont des poids Keras (agent GAT). Ce script
RÉPLIQUE model_test.test et doit être LANCÉ DEPUIS le repo CoLight (config/, models/,
l'env... importables sur le PYTHONPATH).

PRÉREQUIS : patch seed dans ton cityflow_env.py CoLight (reset) :
    "seed": self.dic_traffic_env_conf.get("SEED", 0),     # au lieu de "seed": 0
sinon toutes les seeds renvoient le même résultat.

    python seed_eval_colight.py --records_dir records --model_dir model --dry_run   # vérifie le mapping
    python seed_eval_colight.py --records_dir records --model_dir model --seeds 0 1 2 3 4
    python seed_eval_colight.py --records_dir records --model_dir model --traffic_file autre_flux.json --data_dir data/Hangzhou/4_4
"""

import os
import glob
import json
import csv
import shutil
import tempfile
import argparse
from collections import OrderedDict
import sys

sys.path.append('utils')

import numpy as np
import pandas as pd


# ----------------------------- bookkeeping (tourne partout) -----------------------------
def best_round(run_dir):
    """Round au meilleur avg_reward (moyenné sur les générateurs s'il y en a plusieurs)."""
    df = pd.read_csv(os.path.join(run_dir, "train_reward.csv"))
    g = df.groupby("round")["avg_reward"].mean()
    return int(g.idxmax()), float(g.max())


def discover_configs(records_dir):
    """records/<config>/<run>/train_reward.csv -> {config: [run_dirs]}."""
    configs = OrderedDict()
    for p in sorted(glob.glob(os.path.join(records_dir, "*", "*", "train_reward.csv"))):
        run_dir = os.path.dirname(p)
        config = os.path.relpath(run_dir, records_dir).split(os.sep)[0]
        configs.setdefault(config, []).append(run_dir)
    return configs


def pick_best_run(run_dirs):
    """Parmi les runs d'une config, garde celle dont le meilleur round a le reward le plus haut."""
    best = None
    for rd in run_dirs:
        try:
            rnd, rew = best_round(rd)
        except Exception:
            continue
        if best is None or rew > best[1]:
            best = (rd, rew, rnd)
    return best  # (run_dir, reward, round) | None


def duration_from_log(log_dir, num_inter, run_counts):
    """ATT = duration façon CoLight (summary_detail_RL) à partir des vehicle_inter_*.csv."""
    dfs = []
    for i in range(num_inter):
        df = pd.read_csv(os.path.join(log_dir, f"vehicle_inter_{i}.csv"),
                         sep=",", header=0, dtype={0: str, 1: float, 2: float},
                         names=["vehicle_id", "enter_time", "leave_time"])
        df["leave_time"] = df["leave_time"].fillna(run_counts)
        df["duration"] = df["leave_time"].values - df["enter_time"].values
        df = df[~df["vehicle_id"].astype(str).str.contains("shadow", na=False)]
        dfs.append(df)
    allv = pd.concat(dfs)
    return float(allv.groupby("vehicle_id")["duration"].sum().mean())


# ----------------------------- inférence CoLight (calquée sur model_test.test) -----------
def colight_eval_one(run_dir, model_run_dir, rnd, seed, traffic_file, data_dir):
    """UN épisode de test (round `rnd`, seed `seed`) -> ATT (duration). À lancer DEPUIS le repo CoLight."""
    from config import DIC_AGENTS, DIC_ENVS                       # repo CoLight sur le PYTHONPATH

    dic_agent_conf = json.load(open(os.path.join(run_dir, "agent.conf")))
    dic_exp_conf   = json.load(open(os.path.join(run_dir, "exp.conf")))
    conf = None
    for name in ("anon_env.conf", "sumo_env.conf", "traffic_env.conf"):
        f = os.path.join(run_dir, name)
        if os.path.exists(f):
            conf = json.load(open(f)); break
    if conf is None:
        raise FileNotFoundError("aucun *_env.conf dans " + run_dir)

    agent_name = dic_exp_conf["MODEL_NAME"]
    if agent_name in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:   # éval gloutonne
        dic_agent_conf["EPSILON"] = 0
        dic_agent_conf["MIN_EPSILON"] = 0

    roadnet = conf["ROADNET_FILE"]
    flow = traffic_file or conf["TRAFFIC_FILE"]
    src = data_dir or run_dir                                    # roadnet + flow vivent dans le run_dir
    conf = dict(conf)
    conf["TRAFFIC_FILE"] = flow
    conf["SEED"] = seed                                          # lu par le patch seed de reset()
    conf["IF_GUI"] = False

    tmp = tempfile.mkdtemp()
    work, log = os.path.join(tmp, "work"), os.path.join(tmp, "log")
    os.makedirs(work); os.makedirs(log)
    shutil.copy(os.path.join(src, roadnet), os.path.join(work, roadnet))
    shutil.copy(os.path.join(src, flow), os.path.join(work, flow))

    try:
        env = DIC_ENVS[conf["SIMULATOR_TYPE"]](path_to_log=log,
                                               path_to_work_directory=work,
                                               dic_traffic_env_conf=conf)
        state = env.reset()
        dic_path = {"PATH_TO_MODEL": model_run_dir, "PATH_TO_WORK_DIRECTORY": work}

        if agent_name == "CoLight_Signal":
            agent = DIC_AGENTS[agent_name](dic_agent_conf, conf, dic_path, cnt_round=1,
                                           inter_info=env.list_intersection, intersection_id="0")
        else:
            agent = DIC_AGENTS[agent_name](dic_agent_conf, conf, dic_path, cnt_round=1,
                                           intersection_id="0")

        # CoLight : NUM_AGENTS=1, un réseau partagé -> "round_N_inter_0" (sauf ONE_MODEL)
        if conf.get("ONE_MODEL"):
            agent.load_network(f"round_{rnd}")
        else:
            agent.load_network(f"round_{rnd}_inter_0")

        done = False
        step_num = 0
        n_steps = int(conf["RUN_COUNTS"] / conf["MIN_ACTION_TIME"])
        while not done and step_num < n_steps:
            if "CoLight" in agent_name:
                action_list, _att = agent.choose_action(step_num, state)   # CoLight -> (actions, attention)
            else:
                action_list = agent.choose_action(step_num, state)
            state, _reward, done, _ = env.step(action_list)
            step_num += 1

        env.bulk_log_multi_process()                             # écrit vehicle_inter_*.csv dans `log`
        try:
            env.end_sumo()
        except Exception:
            pass

        return duration_from_log(log, conf["NUM_INTERSECTIONS"], conf["RUN_COUNTS"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records_dir", default="records")
    p.add_argument("--model_dir", default="model", help="dossier model/ au même niveau que records/")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--traffic_file", default=None, help="évaluer sur un autre flux")
    p.add_argument("--data_dir", default=None, help="dossier contenant roadnet+flux si flux externe")
    p.add_argument("--dry_run", action="store_true", help="affiche seulement config -> meilleur round -> .h5")
    p.add_argument("--out", default="seed_eval_colight.csv")
    args = p.parse_args()

    configs = discover_configs(args.records_dir)
    if not configs:
        print(f"Aucune config trouvée sous {args.records_dir}"); return

    rows = []
    for config, run_dirs in configs.items():
        chosen = pick_best_run(run_dirs)
        if chosen is None:
            print(f"[{config}] aucun train_reward lisible — ignoré"); continue
        run_dir, reward, rnd = chosen
        run_name = os.path.basename(run_dir)
        model_run_dir = os.path.join(args.model_dir, config, run_name)
        weight = os.path.join(model_run_dir, f"round_{rnd}_inter_0.h5")
        have_weight = os.path.isfile(weight)

        print(f"=== {config} ===")
        print(f"  run={run_name} | meilleur round={rnd} (avg_reward={reward:.3f})")
        print(f"  poids: {weight} {'(OK)' if have_weight else '(INTROUVABLE !)'}")

        if args.dry_run:
            rows.append({"config": config, "run": run_name, "best_round": rnd,
                         "avg_reward": round(reward, 3), "weight_found": have_weight})
            continue
        if not have_weight:
            print("  -> poids manquants, config sautée"); continue

        atts = []
        for s in args.seeds:
            att = colight_eval_one(run_dir, model_run_dir, rnd, s, args.traffic_file, args.data_dir)
            atts.append(att)
            print(f"    seed {s} -> ATT {att:.2f}")
        mean, std = float(np.mean(atts)), float(np.std(atts))
        print(f"  ==> ATT {mean:.2f} ± {std:.2f}")
        rows.append({"config": config, "run": run_name, "best_round": rnd,
                     "avg_reward": round(reward, 3),
                     "att_mean": round(mean, 2), "att_std": round(std, 2),
                     "atts": ";".join(f"{a:.2f}" for a in atts)})

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nÉcrit : {args.out}")


if __name__ == "__main__":
    main()
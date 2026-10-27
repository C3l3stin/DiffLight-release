#!/usr/bin/env python3
"""DiffLight complexity on an NxM grid, using the repository's actual STFormer.

Copy to DiffLight-release/measure_complexity.py and execute from repository root.
No CityFlow environment, checkpoint, or training is started.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

from models.diffusion.spatial_temoral import STFormer
from models.diffusion.temporal import TemporalUnet
from models.diffusion.diffusion import GaussianInvDynDiffusion


def lane_neighbors(rows, cols, device):
    """Same lane-neighbor topology as DiffLightAgent.get_lane_neighbors."""
    n = rows * cols
    idx = np.zeros((n * 12, 3), dtype=np.int64)
    mask = np.zeros_like(idx)
    for inter in range(n):
        row, col = divmod(inter, cols)
        for lane in range(12):
            group = lane // 3
            neighbor = None
            offsets = None
            if group == 0 and col > 0:
                neighbor, offsets = inter - 1, (1, 5, 9)
            elif group == 2 and col + 1 < cols:
                neighbor, offsets = inter + 1, (3, 7, 11)
            elif group == 1 and row > 0:
                neighbor, offsets = inter - cols, (0, 4, 8)
            elif group == 3 and row + 1 < rows:
                neighbor, offsets = inter + cols, (2, 6, 10)
            if neighbor is not None:
                idx[inter * 12 + lane] = [neighbor * 12 + j for j in offsets]
                mask[inter * 12 + lane] = 1
    return (torch.tensor(idx, device=device), torch.tensor(mask, device=device))


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def flop_lower_bound(fn, device):
    acts = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        acts.append(ProfilerActivity.CUDA)
    with torch.inference_mode():
        with profile(activities=acts, with_flops=True) as prof:
            fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
    return sum(entry.flops or 0 for entry in prof.key_averages())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=4)
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--block-depth', type=int, default=1)
    ap.add_argument('--sample-step', type=int, default=1, help='Sampling stride, 1 means 100 diffusion steps')
    ap.add_argument('--unet', action='store_true')
    ap.add_argument('--drop-neighbor', action='store_true')
    ap.add_argument('--drop-dcm', action='store_true')
    ap.add_argument('--drop-prcd', action='store_true')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--csv', default='complexity_results.csv')
    args = ap.parse_args()
    if args.sample_step <= 0 or args.horizon <= 3:
        ap.error('sample-step must be positive and horizon must be greater than 3')
    if args.device == 'cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested but unavailable')

    device = torch.device(args.device)
    torch.manual_seed(0)
    cond_step = args.horizon - 3  # run_difflight.py
    if args.unet:
        backbone = TemporalUnet(horizon=args.horizon, transition_dim=24,
                                cond_dim=cond_step, dim=64, dim_mults=(1, 2, 4),
                                returns_condition=True, condition_dropout=0.25)
    else:
        backbone = STFormer(horizon=args.horizon, cond_step=cond_step,
                            transition_dim=2, hidden_dim=64,
                            block_depth=args.block_depth, reward_condition=True,
                            condition_dropout=0.25, drop_neighbor=args.drop_neighbor)

    model = GaussianInvDynDiffusion(
        model=backbone, horizon=args.horizon, cond_step=cond_step,
        lane_num=12, observation_dim=2, action_dim=1, eta=0,
        n_timesteps=100, sample_steps=args.sample_step, loss_type='l2',
        clip_denoised=True, predict_epsilon=True, hidden_dim=64,
        action_weight=10, loss_discount=1, loss_weights=None,
        returns_condition=True, condition_guidance_w=1.2, ar_inv=False,
        train_only_inv=False, use_unet=args.unet).to(device).eval()

    n = args.rows * args.cols
    cond = torch.zeros((1, n, cond_step, 12, 2), device=device)
    rewards = torch.zeros((1, n, args.horizon), device=device)
    obs_mask = torch.ones((1, n, args.horizon), device=device)
    neighbor_info = lane_neighbors(args.rows, args.cols, device)

    def complete_decision():
        states = model(cond=cond, returns=rewards, neighbor_info=neighbor_info,
                       obs_mask=obs_mask, drop_dcm=args.drop_dcm,
                       drop_prcd=args.drop_prcd)
        pair = torch.cat([states[..., cond_step - 1, :, :],
                          states[..., cond_step, :, :]], dim=-1)
        scores = model.inv_model(pair.reshape(n, -1))
        return scores.argmax(dim=-1)

    steps = len(range(0, model.n_timesteps, model.sample_steps))
    print(f'Diffusion steps: {steps}')
    print(f'Denoiser calls per decision: {2 * steps} (reward guidance enabled)')
    print('NOTE: torch.profiler reports a LOWER BOUND: unsupported operations are excluded.')
    with torch.inference_mode():
        complete_decision()  # Warm-up
        f = flop_lower_bound(complete_decision, device)
        latency = []
        for _ in range(3):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            complete_decision()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latency.append((time.perf_counter() - start) * 1000)

    record = dict(model='DiffLight', intersections=n, horizon=args.horizon,
                  sample_step=args.sample_step, diffusion_steps=steps,
                  denoiser_calls=2 * steps,
                  parameters_total=count_parameters(model),
                  parameters_denoiser=count_parameters(model.model),
                  parameters_inverse=count_parameters(model.inv_model),
                  gflops_decision_lower_bound=f / 1e9,
                  latency_ms=float(np.median(latency)), device=str(device),
                  unet=args.unet, drop_dcm=args.drop_dcm, drop_prcd=args.drop_prcd)
    for key, value in record.items():
        print(f'{key}: {value}')
    path = Path(args.csv)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    print(f'Saved: {path.resolve()}')


if __name__ == '__main__':
    main()

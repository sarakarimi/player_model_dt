"""
beta (KL weight) ablation for StyleDT (StyleVAEPromptDT).

For each beta we retrain StyleDT from scratch (same hyper-parameters as the main
model, only `beta` changes) and report, all under identical eval settings:

  mean_kl        KL( q(z|tau) || p(z|c) ) averaged over the dataset  (collapse -> 0)
  ctrl_fidelity  mean SIGNED Spearman r between intended and achieved controls
  SR             success rate (reached the goal, any style)
  SAR            style-achievement rate (achieved == target | success)
  D_eff          SAR-weighted trajectory diversity (mean over styles)  (collapse -> low)

Narrative: too-small beta -> loose prior / weaker control; too-large beta ->
posterior collapse (mean_kl -> 0, D_eff drops). The chosen beta should sit in the
sweet spot (high control + healthy diversity).

Run (GPU recommended):
    python style_decision_transformer/utils/beta_ablation.py
    python style_decision_transformer/utils/beta_ablation.py --betas 0 0.0085 0.05 0.2 --epochs 150
"""

import argparse
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from style_decision_transformer.style_pdt_vae.paths import paths
from style_decision_transformer.style_pdt_vae.pdt_vae_with_prior import (
    MiniGridDataset,
    StyleVAEPromptDT,
    train_style_prompt_dt,
)
from style_decision_transformer.style_pdt_vae.control_prompt_pdt import CONTROL_DIM
import style_decision_transformer.utils.evaluate_metrics as em
from style_decision_transformer.utils.evaluate_metrics import (
    StyleVAEAdapter,
    run_full_evaluation,
    set_global_seeds,
)
import style_decision_transformer.utils.eval_diversity_metrics as edm


def diversity_for_model(model, dataset, n_z, div_seeds, max_context):
    """Mean SAR-weighted trajectory diversity across styles (uses eval_diversity)."""
    ctrl_arr = np.asarray(dataset.controls)
    task_arr = np.asarray(dataset.tasks)
    state_mean, state_std = dataset.state_mean, dataset.state_std
    per_style = []
    for sid, sname in edm.STYLE_ORDER.items():
        mask = task_arr == sid
        if not mask.any():
            continue
        c = ctrl_arr[mask].mean(axis=0).astype(np.float32)
        vals = []
        for seed in div_seeds:
            p, ach, _ = edm.collect_group(
                "StyleDT", (model, None), c, sname,
                state_mean, state_std, seed, n_z, 1.0, max_context,
            )
            vals.append(edm.group_metrics(p, ach, sname)["D_traj_eff"])
        per_style.append(float(np.mean(vals)))
    return float(np.mean(per_style)) if per_style else float("nan")


def evaluate_model(model, dataset, device, eval_eps, num_cond, n_z, div_seeds, base_seed):
    max_context = model.dt.max_length
    set_global_seeds(base_seed)
    adapter = StyleVAEAdapter(model, dataset, device=device)
    res = run_full_evaluation(
        adapter, dataset=dataset, device=device,
        num_episodes_per_style=eval_eps, num_conditionings=num_cond,
        max_ep_len=100, initial_rtg=1.0, max_context=max_context, base_seed=base_seed,
    )
    overall = res["rollout"]["overall"]
    cf = res["control_fidelity"]
    signed = [cf[d]["spearman_r"] for d in em.CONTROL_NAMES_ACTIVE
              if d in cf and cf[d].get("spearman_r") is not None]
    mean_signed_rho = float(np.mean(signed)) if signed else float("nan")
    per_dim_rho = {d: cf.get(d, {}).get("spearman_r", float("nan")) for d in em.CONTROL_NAMES_ACTIVE}

    set_global_seeds(base_seed)
    d_eff = diversity_for_model(model, dataset, n_z, div_seeds, max_context)

    return {
        "mean_kl":       res["latent"].get("mean_kl_divergence", float("nan")),
        "ctrl_fidelity": mean_signed_rho,
        "per_dim_rho":   per_dim_rho,
        "SR":            overall.get("success_rate", float("nan")),
        "SAR":           overall.get("style_achievement_rate", float("nan")),
        "D_eff":         d_eff,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.0085, 0.05, 0.2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_episodes", type=int, default=8, help="rollouts per (style, conditioning)")
    ap.add_argument("--num_conditionings", type=int, default=5)
    ap.add_argument("--div_n_z", type=int, default=10)
    ap.add_argument("--div_seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "beta_ablation"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_dir = os.path.join(args.outdir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Device: {device} | betas: {args.betas} | epochs: {args.epochs}")

    max_len = 8
    control_dim = CONTROL_DIM
    dataset = MiniGridDataset(
        trajectory_paths=paths, sampling=True, index_channel_only=True,
        state_normalization_factor=1, action_normalization_factor=1,
        max_len=max_len, control_dim=control_dim,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)
    model_max_ep_len = dataset.max_ep_len + 1

    results = {}
    out_json = os.path.join(args.outdir, "beta_ablation_results.json")

    for beta in args.betas:
        tag = f"{beta:g}"
        print(f"\n{'='*70}\n  TRAINING StyleDT  beta={tag}\n{'='*70}")
        set_global_seeds(args.seed)
        model = StyleVAEPromptDT(
            state_dim=9, act_dim=7, hidden_size=128, latent_dim=16,
            max_length=max_len, max_ep_len=model_max_ep_len, action_tanh=False,
            beta=beta, control_dim=control_dim, prior_hidden=128,
            free_bits=0.0, n_layer=4, n_head=8,
        )
        train_style_prompt_dt(
            model=model, dataloader=loader, num_epochs=args.epochs, device=device,
            lr=args.lr, grad_clip=1.0, action_loss_weight=1.0,
            log_every=0, save_path=None,
            eval_every=10**9,            # skip in-training online eval (final eval below)
            beta_warmup_epochs=0,
        )
        ckpt = os.path.join(ckpt_dir, f"style_beta_{tag}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Saved {ckpt}")

        # Evaluate on CPU (env rollouts are numpy/CPU; eval_diversity assumes CPU).
        model.to("cpu").eval()
        print(f"--- Evaluating beta={tag} ---")
        row = evaluate_model(
            model, dataset, device="cpu",
            eval_eps=args.eval_episodes, num_cond=args.num_conditionings,
            n_z=args.div_n_z, div_seeds=args.div_seeds, base_seed=args.seed,
        )
        results[tag] = row
        print(f"  beta={tag}  mean_kl={row['mean_kl']:.4f}  "
              f"ctrl_fidelity={row['ctrl_fidelity']:.3f}  "
              f"SR={row['SR']:.3f}  SAR={row['SAR']:.3f}  D_eff={row['D_eff']:.3f}")

        # Persist incrementally so partial progress survives.
        with open(out_json, "w") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=2)

    # Final table
    print("\n" + "=" * 78)
    print("BETA ABLATION")
    print(f"{'beta':>8} {'mean_kl':>10} {'ctrl_fid':>10} {'SR':>8} {'SAR':>8} {'D_eff':>8}")
    print("-" * 78)
    for tag, r in results.items():
        print(f"{tag:>8} {r['mean_kl']:>10.4f} {r['ctrl_fidelity']:>10.3f} "
              f"{r['SR']:>8.3f} {r['SAR']:>8.3f} {r['D_eff']:>8.3f}")
    print("=" * 78)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
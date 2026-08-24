# Not Your Average Agent: Play-style Conditioned Decision Transformers

Supplementary code for the paper **"Not Your Average Agent: Play-style Conditioned Decision Transformers"** Proceedings of the 22nd AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE 26).

## About

Offline datasets of human or agent gameplay mix together many different ways of
playing. A policy trained on such a mixture collapses to the "average" player and
cannot be steered towards any particular one. This work conditions a Decision
Transformer on a *play-style* representation:

- A transformer encoder reads a full trajectory and infers a latent style vector
  `z` (a VAE posterior `q(z | trajectory)`).
- `z` is decoded into a short prefix of style tokens that is prepended to the
  Decision Transformer's usual (return-to-go, state, action) token sequence.
- A learned **conditional prior** `p(z | c)` maps a low-dimensional, interpretable
  control vector `c` — computed from per-episode behavioural statistics, e.g.
  *risk taking / stealth exposure / confrontation* — to a distribution over styles.
  At inference time a style can therefore be requested by naming its controls,
  rather than only by supplying a demonstration.

The result is a single model that reproduces distinct, controllable play-styles
rather than an average of them.

Experiments run on custom MiniGrid environments in which several stylistically
distinct routes (e.g. bypassing, using a weapon, using camouflage) all solve the
same task, so style is separable from competence.

## Repository contents

| Path | Contents |
|---|---|
| `envs/` | MiniGrid environments with multiple viable play-styles (`three_`/`four_`/`multi_style_minigrid_env.py`) |
| `ppo/` | PPO training of the per-style expert policies and collection of the offline trajectory datasets |
| `dataset_utils/` | Trajectory datasets and batching for the Decision Transformer / VAE variants |
| `style_decision_transformer/` | StyleDT (ours) — `pdt_vae_with_prior.py` — and the baselines `control_prompt_pdt.py` (ControlDT), `prompt_dt.py` (PromptDT), `bc.py` (BC), `sorl.py` (SORL). `pdt_vae.py` is the earlier variant with a fixed standard-normal prior (no control conditioning) |
| `style_decision_transformer/utils/` | Evaluation and style-controllability metrics (`evaluate_metrics.py`) |
| `datasets/` | Collected offline trajectory datasets |

## Requirements
```
python 3.10.6
```

## Setup
Upgrade pip (we recommend creating a virtual environment for this project)
```
pip install --upgrade pip
```
Install dependencies:
```
pip install -r requirements.txt
```


## Hyperparameters and training details 

**Architecture**

  | Hyperparameter | StyleDT (ours) | ControlDT | PromptDT | BC | SORL |
  |---|---|---|---|---|---|
  | Hidden dim | 128 | 128 | 128 | 256 | 256 |
  | Transformer layers | 4 | 4 | 4 | — | — |
  | Attention heads | 8 | 8 | 8 | — | — |
  | MLP layers | — | — | — | 3 | 3 |
  | Context window | 8 | 8 | 8 | — | — |
  | Prompt length | — | — | 2 | — | — |
  | Latent dim | 16 | — | — | — | — |
  | Prior hidden dim | 128 | — | — | — | — |
  | Control dim | 3 | 3 | — | — | — |

**Training**

 | Hyperparameter | StyleDT (ours) | ControlDT | PromptDT | BC | SORL |
  |---|---|---|---|---|------|
  | Batch size | 32 | 32 | 32 | 256 | 256  |
  | Epochs | 100 | 100 | 100 | 100 | 100  |
  | Learning rate | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
  | Gradient clip | 1.0 | 1.0 | 1.0 | — | 1.0  |

**Method-specific**

| Hyperparameter | StyleDT (ours) | ControlDT | PromptDT | BC | SORL |
|---|---|---|---|---|---|
  | KL weight β | 0.0085 | — | — | — | — |
  | β warmup epochs | 20 | — | — | — | — |
  | EM iterations | — | — | — | — | 30 |
  | M-step epochs/iter | — | — | — | — | 5 |
  | BC warmup epochs | — | — | — | — | 5 |
  | Advantage weight β | — | — | — | — | 1.0 |
  | Advantage clip | — | — | — | — | 5.0 |
  | EM temperature τ | — | — | — | — | 1.0 |
  | Value net epochs | — | — | — | — | 100 |

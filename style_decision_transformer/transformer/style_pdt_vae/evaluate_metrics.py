"""
General evaluation framework for style-conditioned MiniGrid models.

Supports four model types via a ModelAdapter interface:
  - StyleVAEPromptDT  (main approach, pdt_vae_with_prior.py)
  - BCPolicy          (oracle BC baseline, one MLP per style)
  - PromptingDecisionTransformer  (reference-trajectory prompt baseline)
  - ControlConditionedDT          (designer control-vector baseline)

Metrics
-------
Rollout (online, per style):
  success_rate             % of episodes reaching the goal
  style_achievement_rate   % of successes where achieved_style == target_style
  avg_return               mean episode return
  avg_episode_length       mean steps per episode
  detection_rate           % of episodes terminated by detection
  avg_enemy_distance       proxy for stealth / safety behaviour
  avg_path_efficiency      proxy for commitment / directness
  weapon_usage_rate        % of episodes where weapon was picked up
  camouflage_usage_rate    % of episodes where camouflage was picked up

Control fidelity (StyleVAE and ControlDT, requires control_vector per episode):
  mean_abs_spearman_r      mean |Spearman r| between control dims and outcomes
  mean_mse                 mean MSE between input control values and measured outcomes

Latent quality (StyleVAE only, offline from dataset forward pass):
  encoder_silhouette        sklearn silhouette score of encoder z by style label
  encoder_style_accuracy    logistic-regression accuracy z → style (linear probe)
  prior_silhouette          silhouette score of prior z = mu_p(c)
  mean_kl_divergence        KL( q(z|traj) || p(z|c) ) averaged over dataset
  action_accuracy           top-1 action prediction accuracy on the dataset

Usage
-----
Run from repo root:
    python trajectory_embedding/style_dec_vae/transformer/style_pdt_vae/evaluate_metrics.py

Or import and call run_full_evaluation(adapter, dataset) for programmatic use.
"""

import os
import sys
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from style_decision_transformer import paths
from style_decision_transformer.transformer.style_pdt_vae.pdt_vae_with_prior import (
    MiniGridDataset,
    StyleVAEPromptDT,
    kl_q_p_diag,
)
from style_decision_transformer.transformer.style_pdt_vae.bc import BCPolicy
from style_decision_transformer.transformer.style_pdt_vae.prompt_dt import (
    PromptingDecisionTransformer,
)
from style_decision_transformer.transformer.style_pdt_vae.control_prompt_pdt import (
    ControlConditionedDT,
)
from style_decision_transformer.transformer.style_pdt_vae.sorl import (
    SODataset,
    BCPolicy as SORLPolicy,
)
from envs.three_style_env import MiniGridThreeStyles


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STYLE_NAMES  = {0: "bypass", 1: "weapon", 2: "camouflage"}
STYLE_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}

CONTROL_NAMES = [
    "risk_tolerance",
    "resource_pref",
    # "stealth_pref",
    # "safety_pref",
    "commitment",
]

CONTROL_OUTCOME_MAP = [
    (0, "inv_min_dist",       +1),
    (1, "resource_used",      +1),
    # (2, "stealth_score",      +1),
    # (3, "avg_enemy_distance", +1),
    (2, "path_efficiency",    +1),
]
MAX_ENEMY_DIST = 12.0


# ---------------------------------------------------------------------------
# Per-episode data container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    style_id:           int
    target_style:       str
    control_vector:     Optional[np.ndarray]    # None for BC (no explicit control)
    episode_return:     float
    length:             int
    success:            bool
    achieved_style:     Optional[str]
    detected:           bool
    avg_enemy_distance: Optional[float] = None
    min_enemy_distance: Optional[float] = None
    path_efficiency:    Optional[float] = None
    items_picked:       Optional[int]   = None
    picked_weapon:      Optional[bool]  = None
    picked_camouflage:  Optional[bool]  = None


# ---------------------------------------------------------------------------
# ModelAdapter base class
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """
    Uniform interface over all four model types.

    Subclasses implement:
      get_conditionings(style_id, dataset, n) -> list of conditioning objects
      prepare(conditioning, style_id)          -> set up per-episode state
      get_action(states, actions, rtgs, timesteps, attn_mask) -> int
      control_vector() -> np.ndarray | None
      eval() / train()
    """

    name: str = "model"
    uses_control_vectors: bool = False     # True → compute control fidelity
    supports_latent_metrics: bool = False  # True → compute latent / KL metrics

    @abstractmethod
    def get_conditionings(
        self,
        style_id: int,
        dataset:  MiniGridDataset,
        n:        int,
    ) -> list:
        """
        Return up to *n* conditioning objects for the given style.
        Each is passed to prepare() once before a group of episodes.
        For models with no per-style conditioning (BC), return [None].
        """

    @abstractmethod
    def prepare(self, conditioning, style_id: int) -> None:
        """Called once per conditioning before the episode loop starts."""

    @abstractmethod
    def get_action(
        self,
        states:    torch.Tensor,   # [1, T, state_dim]
        actions:   torch.Tensor,   # [1, T, 1]
        rtgs:      torch.Tensor,   # [1, T, 1]
        timesteps: torch.Tensor,   # [1, T]
        attn_mask: torch.Tensor,   # [1, T]
    ) -> int:
        """Return the greedy action for the current timestep."""

    def control_vector(self) -> Optional[np.ndarray]:
        """Return the current conditioning's control vector (for fidelity metrics)."""
        return None

    @property
    def state_mean(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def state_std(self) -> np.ndarray:
        raise NotImplementedError

    def eval(self): pass
    def train(self): pass


# ---------------------------------------------------------------------------
# StyleVAEAdapter
# ---------------------------------------------------------------------------

class StyleVAEAdapter(ModelAdapter):
    """Wraps StyleVAEPromptDT.  Uses prior to sample style tokens from c."""

    name = "StyleVAE"
    uses_control_vectors = True
    supports_latent_metrics = True

    def __init__(
        self,
        model:               StyleVAEPromptDT,
        dataset:             MiniGridDataset,
        device:              str  = "cpu",
        deterministic_prior: bool = False,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.deterministic_prior = deterministic_prior
        self._style_tokens: Optional[torch.Tensor] = None
        self._control_vec:  Optional[np.ndarray]   = None

    @property
    def state_mean(self):
        return self.dataset.state_mean

    @property
    def state_std(self):
        return self.dataset.state_std

    def get_conditionings(self, style_id, dataset, n):
        style_idx = [i for i, t in enumerate(dataset.tasks) if t == style_id]
        if not style_idx:
            return []
        sampled = random.sample(style_idx, min(n, len(style_idx)))
        return [dataset.controls[i] for i in sampled]

    def prepare(self, conditioning, style_id):
        c_np = conditioning
        self._control_vec = c_np.copy()
        c = torch.tensor(c_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            z = self.model.sample_z_from_prior(c, deterministic=self.deterministic_prior)
            self._style_tokens = self.model.latent_to_style_tokens(z)

    def get_action(self, states, actions, rtgs, timesteps, attn_mask):
        with torch.no_grad():
            _, action_preds, _ = self.model.dt.forward(
                states=states, actions=actions, returns_to_go=rtgs,
                timesteps=timesteps, attention_mask=attn_mask,
                style_tokens=self._style_tokens,
            )
        return int(torch.argmax(action_preds[:, -1], dim=-1).item())

    def control_vector(self):
        return self._control_vec

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()


# ---------------------------------------------------------------------------
# BCAdapter
# ---------------------------------------------------------------------------

class BCAdapter(ModelAdapter):
    """
    Wraps a dict of per-style BCPolicy models.
    BC uses only the last state (no temporal context).
    """

    name = "BC"
    uses_control_vectors = False
    supports_latent_metrics = False

    def __init__(
        self,
        policies:    Dict[int, BCPolicy],
        state_mean:  np.ndarray,
        state_std:   np.ndarray,
        device:      str = "cpu",
    ):
        self.policies = policies
        self._state_mean = state_mean
        self._state_std  = state_std
        self.device = device
        self._policy: Optional[BCPolicy] = None

    @property
    def state_mean(self):
        return self._state_mean

    @property
    def state_std(self):
        return self._state_std

    def get_conditionings(self, style_id, dataset, n):
        return [None]   # BC selects policy by style_id; no per-conditioning variability

    def prepare(self, conditioning, style_id):
        self._policy = self.policies[style_id]

    def get_action(self, states, actions, rtgs, timesteps, attn_mask):
        last_state = states[0, -1]   # [state_dim]
        with torch.no_grad():
            logits = self._policy.forward(last_state.unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())

    def eval(self):
        for p in self.policies.values():
            p.eval()

    def train(self):
        for p in self.policies.values():
            p.train()


# ---------------------------------------------------------------------------
# PromptDTAdapter
# ---------------------------------------------------------------------------

class PromptDTAdapter(ModelAdapter):
    """
    Wraps PromptingDecisionTransformer.
    Samples one reference trajectory from the dataset as the prompt.

    The prompt is always exactly `prompt_length` steps long (matching how the
    model was trained in PromptDataset.get_prompt_traj):
      - A random starting index si in [0, max(0, traj_len - prompt_length)] is
        drawn, and `prompt_length` consecutive steps are taken from si.
      - If the trajectory is shorter than prompt_length, the result is
        zero-padded on the left and the attention mask marks those as 0.
    """

    name = "PromptDT"
    uses_control_vectors = False
    supports_latent_metrics = False

    def __init__(
        self,
        model:         PromptingDecisionTransformer,
        dataset:       MiniGridDataset,
        device:        str = "cpu",
        prompt_length: int = 2,    # must match max_len used during PromptDT training
        top_k_returns: int = 2,    # number of top unique return values in prompt pool
        n_per_return:  int = 3,    # trajectories sampled per return level
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.prompt_length = prompt_length
        self._prompt: Optional[Tuple] = None

        # Build prompt pool: top-return trajectories per style,
        # matching PromptDataset._build_prompt_pool().
        # self._prompt_pool = self._build_prompt_pool(dataset, top_k_returns, n_per_return)

    @staticmethod
    def _build_prompt_pool(dataset, top_k_returns: int, n_per_return: int) -> dict:
        style_to_indices: dict = {}
        for idx in dataset.indices:
            label = int(dataset.tasks[idx])
            style_to_indices.setdefault(label, []).append(idx)

        pool = {}
        for style, idxs in style_to_indices.items():
            traj_returns = [(i, float(dataset.rewards[i].sum())) for i in idxs]
            unique_rets  = sorted(set(r for _, r in traj_returns), reverse=True)
            top_rets     = unique_rets[:top_k_returns]

            selected = []
            for ret in top_rets:
                matching = [i for i, r in traj_returns if r == ret]
                selected.extend(random.sample(matching, min(n_per_return, len(matching))))

            pool[style] = selected if selected else idxs  # fallback: use all
        return pool

    @property
    def state_mean(self):
        return self.dataset.state_mean

    @property
    def state_std(self):
        return self.dataset.state_std

    def get_conditionings(self, style_id, dataset, n):
        # candidates = self._prompt_pool.get(style_id, [])
        # if not candidates:
        #     return []
        # return random.sample(candidates, min(n, len(candidates)))  # traj indices
        style_idx = [i for i, t in enumerate(dataset.tasks) if t == style_id]
        if not style_idx:
            return []
        return random.sample(style_idx, min(n, len(style_idx)))  # traj indices

    def prepare(self, conditioning, style_id):
        """
        Build a fixed-length prompt 7-tuple from a reference trajectory.
        Mirrors PromptDataset.get_prompt_traj(): random si in [0, traj_len -
        prompt_length], then take prompt_length steps, zero-pad if short.
        """
        traj_i = conditioning
        PL = self.prompt_length

        def _to_np(x):
            return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

        s_raw = _to_np(self.dataset.states[traj_i])
        a_raw = _to_np(self.dataset.actions[traj_i])
        r_raw = _to_np(self.dataset.rewards[traj_i])
        d_raw = _to_np(self.dataset.dones[traj_i])
        traj_len = len(s_raw)

        # Random start index (same logic as get_prompt_traj)
        si_max = max(0, traj_len - PL)
        si     = random.randint(0, si_max)
        end    = si + PL

        s_slice = s_raw[si:end].reshape(-1, *s_raw.shape[1:]).astype(np.float32)
        a_slice = a_raw[si:end].reshape(-1).astype(np.int64)
        r_slice = r_raw[si:end].reshape(-1, 1).astype(np.float32)
        d_slice = d_raw[si:end].reshape(-1).astype(np.float32)
        ti_slice = np.arange(si, si + len(s_slice), dtype=np.int64)

        # RTG: cumulative sum from each step to end of trajectory
        rewards_full = _to_np(self.dataset.rewards[traj_i]).reshape(-1)
        rtg_vals = np.cumsum(rewards_full[si:][::-1])[::-1]
        rtg_slice = rtg_vals[:len(s_slice)].reshape(-1, 1).astype(np.float32)

        actual_len = len(s_slice)
        pad = PL - actual_len

        # Zero-pad on the left (matching add_padding convention)
        def _pad_left(arr, pad_val, pad_rows):
            if pad_rows == 0:
                return arr
            pad_shape = (pad_rows,) + arr.shape[1:]
            return np.concatenate([np.full(pad_shape, pad_val, dtype=arr.dtype), arr], axis=0)

        s_p  = _pad_left(s_slice.reshape(actual_len, -1), 0,   pad)
        a_p  = _pad_left(a_slice.reshape(actual_len),     0,   pad)   # -10 in training but 0 avoids embed OOB
        r_p  = _pad_left(r_slice,                         0,   pad)
        d_p  = _pad_left(d_slice.reshape(actual_len),     2,   pad)
        rtg_p = _pad_left(rtg_slice,                      0,   pad)
        ti_p = _pad_left(ti_slice.reshape(actual_len),    0,   pad)
        m_p  = np.concatenate([np.zeros(pad, dtype=np.float32), np.ones(actual_len, dtype=np.float32)])

        # Normalise states
        s_p = (s_p - self.dataset.state_mean) / self.dataset.state_std

        dev = self.device
        self._prompt = (
            torch.tensor(s_p,  dtype=torch.float32, device=dev).unsqueeze(0),   # [1, PL, state_dim]
            torch.tensor(a_p,  dtype=torch.long,    device=dev).unsqueeze(0),   # [1, PL]
            torch.tensor(r_p,  dtype=torch.float32, device=dev).unsqueeze(0),   # [1, PL, 1]
            torch.tensor(d_p,  dtype=torch.float32, device=dev).unsqueeze(0),   # [1, PL]
            torch.tensor(rtg_p,dtype=torch.float32, device=dev).unsqueeze(0),   # [1, PL, 1]
            torch.tensor(ti_p, dtype=torch.long,    device=dev).unsqueeze(0),   # [1, PL]
            torch.tensor(m_p,  dtype=torch.float32, device=dev).unsqueeze(0),   # [1, PL]
        )

    def get_action(self, states, actions, rtgs, timesteps, attn_mask):
        with torch.no_grad():
            _, action_preds, _ = self.model.forward(
                states=states,
                actions=actions.squeeze(-1),
                rewards=None,
                returns_to_go=rtgs,
                timesteps=timesteps,
                attention_mask=attn_mask,
                prompt=self._prompt,
            )
        return int(torch.argmax(action_preds[:, -1], dim=-1).item())

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()


# ---------------------------------------------------------------------------
# ControlDTAdapter
# ---------------------------------------------------------------------------

class ControlDTAdapter(ModelAdapter):
    """
    Wraps ControlConditionedDT.
    Uses per-trajectory control vectors sampled from the dataset.
    """

    name = "ControlDT"
    uses_control_vectors = True
    supports_latent_metrics = False

    def __init__(
        self,
        model:   ControlConditionedDT,
        dataset: MiniGridDataset,
        device:  str = "cpu",
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self._control_vec:  Optional[np.ndarray]   = None
        self._ctrl_tensor:  Optional[torch.Tensor] = None

    @property
    def state_mean(self):
        return self.dataset.state_mean

    @property
    def state_std(self):
        return self.dataset.state_std

    def get_conditionings(self, style_id, dataset, n):
        style_idx = [i for i, t in enumerate(dataset.tasks) if t == style_id]
        if not style_idx:
            return []
        sampled = random.sample(style_idx, min(n, len(style_idx)))
        return [dataset.controls[i] for i in sampled]

    def prepare(self, conditioning, style_id):
        c_np = conditioning
        self._control_vec = c_np.copy()
        self._ctrl_tensor = torch.tensor(
            c_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)   # [1, control_dim]

    def get_action(self, states, actions, rtgs, timesteps, attn_mask):
        with torch.no_grad():
            _, action_preds, _ = self.model.forward(
                states=states,
                actions=actions.squeeze(-1),
                rewards=None,
                returns_to_go=rtgs,
                timesteps=timesteps,
                controls=self._ctrl_tensor,
                attention_mask=attn_mask,
            )
        return int(torch.argmax(action_preds[:, -1], dim=-1).item())

    def control_vector(self):
        return self._control_vec

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()


# ---------------------------------------------------------------------------
# SOAdapter
# ---------------------------------------------------------------------------

class SOAdapter(ModelAdapter):
    """
    Wraps K SORL BCPolicy models (one per discovered style).
    Uses style_map to align discovered cluster index → ground-truth style id.
    No per-episode conditioning — each style gets one fixed policy.
    """

    name = "SORL"
    uses_control_vectors = False
    supports_latent_metrics = False

    def __init__(
        self,
        policies:   Dict[int, SORLPolicy],
        so_dataset: SODataset,
        device:     str = "cpu",
        style_map:  Optional[Dict[int, int]] = None,
    ):
        """
        policies:   {discovered_cluster_k: SORLPolicy}
        so_dataset: SODataset (provides state_mean / state_std)
        style_map:  {ground_truth_style_id -> discovered_cluster_k}
                    Defaults to identity. Override once you know the cluster alignment
                    from print_cluster_composition().
        """
        self.policies   = policies
        self.so_dataset = so_dataset
        self.device     = device
        # style_map maps evaluation style_id → which SORL cluster to use
        self._style_map = style_map if style_map is not None else {k: k for k in range(len(policies))}
        self._policy: Optional[SORLPolicy] = None

    @property
    def state_mean(self):
        return self.so_dataset.state_mean

    @property
    def state_std(self):
        return self.so_dataset.state_std

    def get_conditionings(self, style_id, dataset, n):
        return [None]   # no per-episode conditioning

    def prepare(self, conditioning, style_id):
        cluster_k = self._style_map.get(style_id, style_id)
        self._policy = self.policies[cluster_k]

    def get_action(self, states, actions, rtgs, timesteps, attn_mask):
        last_state = states[0, -1]   # [state_dim] — memoryless policy
        with torch.no_grad():
            logits = self._policy.forward(last_state.unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())

    def eval(self):
        for p in self.policies.values():
            p.eval()

    def train(self):
        for p in self.policies.values():
            p.train()


# ---------------------------------------------------------------------------
# Generic rollout
# ---------------------------------------------------------------------------

def _rollout_episode(
    adapter:     ModelAdapter,
    env:         MiniGridThreeStyles,
    state_mean:  torch.Tensor,
    state_std:   torch.Tensor,
    device:      str,
    initial_rtg: float,
    max_ep_len:  int,
    max_context: int,
    seed:        int,
) -> dict:
    """Single episode rollout using a ModelAdapter; returns raw outcome dict."""
    obs, _ = env.reset(seed=seed)

    state = torch.from_numpy(obs["image"][:, :, 0].flatten()).float().to(device)
    state = (state - state_mean) / state_std

    states    = state.reshape(1, 1, -1)
    actions   = torch.zeros((1, 1, 1), dtype=torch.long, device=device)
    rtgs      = torch.tensor([[[initial_rtg]]], dtype=torch.float32, device=device)
    timesteps = torch.tensor([[0]], dtype=torch.long, device=device)

    ep_return = 0.0
    done      = False
    t         = 0
    info      = {}
    env.reset()  # ensure env can render for episode summary
    while not done and t < max_ep_len:
        attn_mask = torch.ones((1, states.shape[1]), dtype=torch.float32, device=device)
        action = adapter.get_action(states, actions, rtgs, timesteps, attn_mask)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_return += float(reward)
        t += 1

        if not done:
            ns = torch.from_numpy(next_obs["image"][:, :, 0].flatten()).float().to(device)
            ns = (ns - state_mean) / state_std
            states    = torch.cat([states,    ns.reshape(1, 1, -1)], dim=1)
            actions   = torch.cat([actions,   torch.tensor([[[action]]], dtype=torch.long, device=device)], dim=1)
            rtgs      = torch.cat([rtgs,      rtgs[:, -1:] - reward], dim=1)
            timesteps = torch.cat([timesteps, torch.tensor([[t]], dtype=torch.long, device=device)], dim=1)
            if max_context and states.shape[1] > max_context:
                states    = states[:, -max_context:]
                actions   = actions[:, -max_context:]
                rtgs      = rtgs[:, -max_context:]
                timesteps = timesteps[:, -max_context:]


    es       = info.get("episode_summary")
    detected = info.get("detected", False) or info.get("termination") == "detected_front_tile"
    success  = (es is not None)

    return dict(
        episode_return=ep_return,
        length=t,
        success=success,
        achieved_style=es.get("achieved_style") if es else None,
        detected=detected,
        avg_enemy_distance=es.get("avg_enemy_distance") if es else None,
        min_enemy_distance=es.get("min_enemy_distance") if es else None,
        path_efficiency=es.get("path_efficiency") if es else None,
        items_picked=es.get("items_picked") if es else None,
        picked_weapon=es.get("picked_weapon") if es else None,
        picked_camouflage=es.get("picked_camouflage") if es else None,
    )


def run_rollout_evaluation(
    adapter:                ModelAdapter,
    dataset:                MiniGridDataset,
    num_episodes_per_style: int   = 5,
    device:                 str   = "cpu",
    initial_rtg:            float = 1.0,
    max_ep_len:             int   = 100,
    max_context:            int   = 20,
    env_kwargs:             dict  = None,
    num_conditionings:      int   = 5,
    base_seed:              int   = 0,
) -> List[EpisodeRecord]:
    """
    For each style, sample `num_conditionings` conditioning objects, call
    adapter.prepare(), then run `num_episodes_per_style` rollouts.
    Returns a flat list of EpisodeRecord objects.
    """
    if env_kwargs is None:
        env_kwargs = {}

    adapter.eval()
    state_mean = torch.tensor(adapter.state_mean, device=device, dtype=torch.float32)
    state_std  = torch.tensor(adapter.state_std,  device=device, dtype=torch.float32)

    records: List[EpisodeRecord] = []
    seed_counter = base_seed * 10_000  # offset per run to avoid seed overlap

    with torch.no_grad():
        for style_id, style_name in STYLE_NAMES.items():
            conditionings = adapter.get_conditionings(style_id, dataset, num_conditionings)
            if not conditionings:
                print(f"  Warning: no conditionings for style {style_name}")
                continue

            for cond in conditionings:
                adapter.prepare(cond, style_id)
                ctrl = adapter.control_vector()

                env = MiniGridThreeStyles(
                    target_style=style_name,
                    target_bonus=1.0,
                    non_target_penalty=-1.0,
                    easy_env=False,
                    agent_view_size=3,
                    randomize_layout=True,
                    **env_kwargs,
                )

                for _ in range(num_episodes_per_style):
                    outcome = _rollout_episode(
                        adapter, env,
                        state_mean, state_std, device,
                        initial_rtg, max_ep_len, max_context,
                        seed=seed_counter,
                    )
                    seed_counter += 1
                    records.append(EpisodeRecord(
                        style_id=style_id,
                        target_style=style_name,
                        control_vector=ctrl.copy() if ctrl is not None else None,
                        **outcome,
                    ))
                env.close()

    adapter.train()
    return records


# ---------------------------------------------------------------------------
# Aggregate rollout metrics
# ---------------------------------------------------------------------------

def aggregate_rollout_metrics(records: List[EpisodeRecord]) -> dict:
    def _agg(recs):
        n = len(recs)
        if n == 0:
            return {}
        successes = [r for r in recs if r.success]
        n_suc     = len(successes)
        achieved  = [r for r in successes if r.achieved_style == r.target_style]

        def _mf(lst, field):
            vals = [getattr(r, field) for r in lst if getattr(r, field) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "n_episodes":             n,
            "success_rate":           n_suc / n,
            "style_achievement_rate": len(achieved) / n_suc if n_suc > 0 else float("nan"),
            "avg_return":             float(np.mean([r.episode_return for r in recs])),
            "avg_episode_length":     float(np.mean([r.length for r in recs])),
            "detection_rate":         float(np.mean([r.detected for r in recs])),
            "avg_enemy_distance":     _mf(successes, "avg_enemy_distance"),
            "avg_path_efficiency":    _mf(successes, "path_efficiency"),
            "weapon_usage_rate":      float(np.mean([r.picked_weapon     for r in successes])) if successes else float("nan"),
            "camouflage_usage_rate":  float(np.mean([r.picked_camouflage for r in successes])) if successes else float("nan"),
        }

    per_style = {
        sname: _agg([r for r in records if r.style_id == sid])
        for sid, sname in STYLE_NAMES.items()
    }
    return {"per_style": per_style, "overall": _agg(records)}


# ---------------------------------------------------------------------------
# Control fidelity
# ---------------------------------------------------------------------------

def compute_control_fidelity(records: List[EpisodeRecord]) -> dict:
    suc = [r for r in records
           if r.success
           and r.control_vector is not None
           and r.avg_enemy_distance is not None
           and r.path_efficiency is not None
           and r.items_picked is not None]

    if len(suc) < 5:
        print("  Warning: too few successful episodes for control fidelity.")
        return {"mean_abs_spearman_r": float("nan"), "per_dim": {}}

    outcomes = {
        "inv_min_dist":       np.array([1.0 - min(r.min_enemy_distance / MAX_ENEMY_DIST, 1.0) for r in suc]),
        "resource_used":      np.clip(np.array([r.items_picked / 2.0 for r in suc]), 0, 1),
        "stealth_score":      np.clip(
                                  np.array([r.avg_enemy_distance / MAX_ENEMY_DIST for r in suc])
                                  * np.array([1.0 - float(r.detected) for r in suc]),
                              0, 1),
        "avg_enemy_distance": np.clip(np.array([r.avg_enemy_distance / MAX_ENEMY_DIST for r in suc]), 0, 1),
        "path_efficiency":    np.array([r.path_efficiency for r in suc]),
    }

    results = {}
    rs = []
    mses = []
    for dim_idx, outcome_key, _ in CONTROL_OUTCOME_MAP:
        ctrl_vals = np.array([r.control_vector[dim_idx] for r in suc])
        outcome_vals = outcomes[outcome_key]
        r_val, p_val = spearmanr(ctrl_vals, outcome_vals)
        mse_val = float(np.mean((ctrl_vals - outcome_vals) ** 2))
        dim_name = CONTROL_NAMES[dim_idx]
        results[dim_name] = {
            "spearman_r": float(r_val),
            "p_value":    float(p_val),
            "mse":        mse_val,
        }
        rs.append(abs(float(r_val)))
        mses.append(mse_val)

    results["mean_abs_spearman_r"] = float(np.mean(rs))
    results["mean_mse"] = float(np.mean(mses))
    return results


# ---------------------------------------------------------------------------
# Latent quality (StyleVAE only, offline)
# ---------------------------------------------------------------------------

def compute_latent_metrics(
    model:      StyleVAEPromptDT,
    dataset:    MiniGridDataset,
    device:     str = "cpu",
    batch_size: int = 64,
) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    all_z_enc   = []
    all_z_prior = []
    all_labels  = []
    all_kl      = []
    n_correct   = 0
    n_total     = 0

    with torch.no_grad():
        for batch in loader:
            full_states    = batch["full_states"].to(device)
            full_actions   = batch["full_actions"].to(device)
            full_timesteps = batch["full_timesteps"].to(device)
            full_mask      = batch["full_attention_mask"].to(device)
            controls       = batch["controls"].to(device)
            states         = batch["states"].to(device)
            actions        = batch["actions"].to(device)
            rtgs           = batch["returns_to_go"].to(device)
            timesteps      = batch["timesteps"].to(device)
            attn_mask      = batch["attention_mask"].to(device)
            labels         = batch["task_labels"]

            mu_q, logvar_q, z_enc = model.encode_full_trajectory(
                full_states, full_actions, full_timesteps, full_mask
            )
            mu_p, logvar_p = model.prior(controls)
            z_prior = mu_p

            kl = kl_q_p_diag(mu_q, logvar_q, mu_p, logvar_p)
            all_kl.extend(kl.cpu().numpy().tolist())

            style_tokens = model.latent_to_style_tokens(z_enc)
            _, action_preds, _ = model.dt(
                states=states, actions=actions, returns_to_go=rtgs,
                timesteps=timesteps, attention_mask=attn_mask,
                style_tokens=style_tokens,
            )
            B, T, C = action_preds.shape
            acts_ce   = torch.clamp(actions.squeeze(-1).long(), 0, C - 1)
            predicted = torch.argmax(action_preds, dim=-1)
            mask_bool = attn_mask.bool()
            n_correct += (predicted[mask_bool] == acts_ce[mask_bool]).sum().item()
            n_total   += mask_bool.sum().item()

            all_z_enc.extend(z_enc.cpu().numpy())
            all_z_prior.extend(z_prior.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

    Z_enc   = np.array(all_z_enc)
    Z_prior = np.array(all_z_prior)
    Y       = np.array(all_labels)

    results = {}
    results["encoder_silhouette"] = float(silhouette_score(Z_enc, Y))
    results["prior_silhouette"]   = float(silhouette_score(Z_prior, Y))

    scaler = StandardScaler()
    Z_sc   = scaler.fit_transform(Z_enc)
    clf    = LogisticRegression(max_iter=1000, random_state=0)
    n      = len(Y)
    split  = int(0.8 * n)
    idx    = np.random.permutation(n)
    clf.fit(Z_sc[idx[:split]], Y[idx[:split]])
    results["encoder_style_accuracy"] = float(clf.score(Z_sc[idx[split:]], Y[idx[split:]]))

    results["mean_kl_divergence"] = float(np.mean(all_kl))
    results["action_accuracy"]    = float(n_correct / n_total) if n_total > 0 else float("nan")

    model.train()
    return results


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def run_full_evaluation(
    adapter:                ModelAdapter,
    dataset:                MiniGridDataset,
    device:                 str   = "cpu",
    num_episodes_per_style: int   = 30,
    num_conditionings:      int   = 5,
    max_ep_len:             int   = 100,
    initial_rtg:            float = 1.0,
    max_context:            int   = 20,
    env_kwargs:             dict  = None,
    base_seed:              int   = 0,
) -> dict:
    """
    Run all applicable metrics for the given adapter and return a unified dict.

    Always runs:  rollout metrics
    Conditional:  control fidelity (if adapter.uses_control_vectors)
                  latent metrics   (if adapter.supports_latent_metrics)
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {adapter.name}")
    print(f"{'='*60}")

    print("=== Rollout evaluation ===")
    records = run_rollout_evaluation(
        adapter, dataset,
        num_episodes_per_style=num_episodes_per_style,
        num_conditionings=num_conditionings,
        device=device,
        initial_rtg=initial_rtg,
        max_ep_len=max_ep_len,
        max_context=max_context,
        env_kwargs=env_kwargs,
        base_seed=base_seed,
    )
    rollout = aggregate_rollout_metrics(records)

    result = {"rollout": rollout, "control_fidelity": {}, "latent": {}}

    if adapter.uses_control_vectors:
        print("=== Control fidelity ===")
        result["control_fidelity"] = compute_control_fidelity(records)

    if adapter.supports_latent_metrics:
        print("=== Latent / reconstruction metrics ===")
        result["latent"] = compute_latent_metrics(adapter.model, dataset, device=device)

    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict, model_name: str = "Model"):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {model_name}")
    print(bar)

    print("\n[ Rollout metrics ]")
    header = f"{'metric':<30} {'overall':>10}" + "".join(f"  {n:>12}" for n in STYLE_NAMES.values())
    print(header)
    print("-" * len(header))
    for k in [
        "success_rate", "style_achievement_rate", "avg_return",
        "avg_episode_length", "detection_rate",
        "avg_enemy_distance", "avg_path_efficiency",
        "weapon_usage_rate", "camouflage_usage_rate",
    ]:
        ov  = metrics["rollout"]["overall"].get(k, float("nan"))
        row = f"{k:<30} {ov:>10.3f}"
        for sname in STYLE_NAMES.values():
            val = metrics["rollout"]["per_style"].get(sname, {}).get(k, float("nan"))
            row += f"  {val:>12.3f}"
        print(row)

    if metrics.get("control_fidelity"):
        print("\n[ Control fidelity  (Spearman r, higher = controls drive behaviour) ]")
        cf = metrics["control_fidelity"]
        for dim_name in CONTROL_NAMES:
            info = cf.get(dim_name, {})
            r    = info.get("spearman_r", float("nan"))
            p    = info.get("p_value",    float("nan"))
            mse  = info.get("mse",        float("nan"))
            sig  = "*" if p < 0.05 else " "
            print(f"  {dim_name:<22}  r = {r:+.3f}  (p={p:.3f}){sig}  mse = {mse:.4f}")
        print(f"  {'mean |r|':<22}  {cf.get('mean_abs_spearman_r', float('nan')):.3f}")
        print(f"  {'mean mse':<22}  {cf.get('mean_mse', float('nan')):.4f}")

    if metrics.get("latent"):
        print("\n[ Latent / reconstruction metrics ]")
        for k, v in metrics["latent"].items():
            print(f"  {k:<35}  {v:.4f}")

    print(f"\n{bar}\n")


def print_comparison_table(all_metrics: dict):
    """all_metrics: {model_name: metrics_dict}"""
    key_metrics = [
        ("success_rate",          lambda m: m["rollout"]["overall"].get("success_rate",           float("nan"))),
        ("style_achievement_rate",lambda m: m["rollout"]["overall"].get("style_achievement_rate", float("nan"))),
        ("avg_return",            lambda m: m["rollout"]["overall"].get("avg_return",             float("nan"))),
        ("detection_rate",        lambda m: m["rollout"]["overall"].get("detection_rate",         float("nan"))),
        ("control_fidelity_|r|",  lambda m: m["control_fidelity"].get("mean_abs_spearman_r",     float("nan"))),
        ("control_fidelity_mse",  lambda m: m["control_fidelity"].get("mean_mse",               float("nan"))),
        ("enc_silhouette",        lambda m: m["latent"].get("encoder_silhouette",                float("nan"))),
        ("enc_style_accuracy",    lambda m: m["latent"].get("encoder_style_accuracy",            float("nan"))),
        ("mean_kl",               lambda m: m["latent"].get("mean_kl_divergence",               float("nan"))),
        ("action_accuracy",       lambda m: m["latent"].get("action_accuracy",                   float("nan"))),
    ]

    model_names = list(all_metrics.keys())
    col_w = max(20, max(len(n) for n in model_names))
    print("\n" + "=" * (34 + col_w * len(model_names)))
    print(f"{'metric':<34}" + "".join(f"{n:>{col_w}}" for n in model_names))
    print("-" * (34 + col_w * len(model_names)))
    for label, extractor in key_metrics:
        row = f"{label:<34}"
        for mn in model_names:
            val = extractor(all_metrics[mn])
            row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * (34 + col_w * len(model_names)) + "\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_comparison(all_metrics: dict, save_dir: str = None, std_metrics: dict = None):
    """
    all_metrics: {model_name: metrics_dict}

    Produces three figures, each with all models in every plot:

      comparison_rollout.png          2×3 grid; one subplot per rollout metric,
                                      grouped bars per style, one colour per model.
      comparison_control_fidelity.png Spearman r per control dim; one bar group
                                      per control dim, one colour per model
                                      (only for models with control fidelity data).
      comparison_latent.png           Latent / reconstruction metrics; one bar group
                                      per metric, one colour per model
                                      (NaN bars shown as empty with a dashed outline).
    """
    if save_dir is None:
        save_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

    model_names = list(all_metrics.keys())
    n_models    = len(model_names)
    palette     = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    ]

    def _bar_label(ax, bar, val, fmt=".2f"):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=7)

    # -------------------------------------------------------------------------
    # Figure 1 — rollout metrics (2 rows × 3 cols)
    # Each subplot: x = styles + "overall", grouped bars per model
    # -------------------------------------------------------------------------
    rollout_show = [
        ("success_rate",           "Success rate"),
        ("style_achievement_rate", "Style achievement rate"),
        ("avg_return",             "Avg return"),
        ("detection_rate",         "Detection rate"),
        ("avg_enemy_distance",     "Avg enemy distance"),
        ("avg_path_efficiency",    "Avg path efficiency"),
    ]

    style_names  = list(STYLE_NAMES.values())
    x_labels     = [s.capitalize() for s in style_names] + ["Overall"]
    n_groups     = len(x_labels)
    x            = np.arange(n_groups)
    width        = 0.8 / n_models

    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
    fig1.suptitle("Model Comparison — Rollout Metrics", fontsize=14, fontweight="bold")

    for ax, (key, title) in zip(axes1.flatten(), rollout_show):
        for i, mn in enumerate(model_names):
            per_style = [
                all_metrics[mn]["rollout"]["per_style"].get(sn, {}).get(key, float("nan"))
                for sn in style_names
            ]
            overall = all_metrics[mn]["rollout"]["overall"].get(key, float("nan"))
            vals   = per_style + [overall]
            offset = (i - n_models / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width=width * 0.9,
                            color=palette[i % len(palette)], alpha=0.85, label=mn)
            for bar, val in zip(bars, vals):
                _bar_label(ax, bar, val)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

    # shared legend under the figure
    handles, labels = axes1.flatten()[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc="lower center", ncol=n_models,
                fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    p1 = os.path.join(save_dir, "plots", "comparison_rollout.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"Saved {p1}")
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2 — control fidelity (Spearman r per dim, grouped by model)
    # Only models that actually computed control fidelity are shown.
    # -------------------------------------------------------------------------
    cf_models = [mn for mn in model_names
                 if all_metrics[mn]["control_fidelity"].get("mean_abs_spearman_r") is not None
                 and not np.isnan(all_metrics[mn]["control_fidelity"].get("mean_abs_spearman_r", float("nan")))]

    if cf_models:
        n_cf     = len(cf_models)
        n_dims   = len(CONTROL_NAMES)
        xc       = np.arange(n_dims)
        wc       = 0.8 / n_cf
        dim_lbls = [d.replace("_", "\n") for d in CONTROL_NAMES]

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.suptitle("Model Comparison — Control Fidelity (Spearman r)", fontsize=13, fontweight="bold")

        for i, mn in enumerate(cf_models):
            cf     = all_metrics[mn]["control_fidelity"]
            r_vals = [cf.get(d, {}).get("spearman_r", float("nan")) for d in CONTROL_NAMES]
            r_stds = None
            if std_metrics is not None and mn in std_metrics:
                cf_s   = std_metrics[mn]["control_fidelity"]
                r_stds = [cf_s.get(d, {}).get("spearman_r", 0.0) for d in CONTROL_NAMES]
            offset = (i - n_cf / 2 + 0.5) * wc
            bars   = ax2.bar(xc + offset, r_vals, width=wc * 0.9,
                             color=palette[model_names.index(mn) % len(palette)],
                             alpha=0.85, label=f"{mn} (|r|={cf.get('mean_abs_spearman_r', float('nan')):.2f})",
                             yerr=r_stds, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"})
            for bar, val in zip(bars, r_vals):
                _bar_label(ax2, bar, val, fmt="+.2f")

        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xticks(xc)
        ax2.set_xticklabels(dim_lbls, fontsize=14)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Spearman r", fontsize=14)
        ax2.yaxis.grid(True, alpha=0.35)
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=9)

        plt.tight_layout()
        p2 = os.path.join(save_dir, "plots", "comparison_control_fidelity.png")
        plt.savefig(p2, dpi=150)
        print(f"Saved {p2}")
        plt.close()

        # MSE plot — same layout as Spearman
        fig2b, ax2b = plt.subplots(figsize=(10, 5))
        fig2b.suptitle("Model Comparison — Control Fidelity (MSE)", fontsize=14, fontweight="bold")

        for i, mn in enumerate(cf_models):
            cf      = all_metrics[mn]["control_fidelity"]
            mse_vals = [cf.get(d, {}).get("mse", float("nan")) for d in CONTROL_NAMES]
            mse_stds = None
            if std_metrics is not None and mn in std_metrics:
                cf_s     = std_metrics[mn]["control_fidelity"]
                mse_stds = [cf_s.get(d, {}).get("mse", 0.0) for d in CONTROL_NAMES]
            offset = (i - n_cf / 2 + 0.5) * wc
            bars   = ax2b.bar(xc + offset, mse_vals, width=wc * 0.9,
                              color=palette[model_names.index(mn) % len(palette)],
                              alpha=0.85, label=f"{mn} (mse={cf.get('mean_mse', float('nan')):.3f})",
                              yerr=mse_stds, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"})
            for bar, val in zip(bars, mse_vals):
                _bar_label(ax2b, bar, val, fmt=".3f")

        ax2b.set_xticks(xc)
        ax2b.set_xticklabels(dim_lbls, fontsize=14)
        ax2b.set_ylabel("MSE (lower = better)", fontsize=14)
        ax2b.yaxis.grid(True, alpha=0.35)
        ax2b.set_axisbelow(True)
        ax2b.legend(fontsize=9)

        plt.tight_layout()
        p2b = os.path.join(save_dir, "plots", "comparison_control_fidelity_mse.png")
        plt.savefig(p2b, dpi=150)
        print(f"Saved {p2b}")
        plt.close()

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

EVAL_SEEDS = [0, 1, 2, 3, 4]


def set_global_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _recursive_aggregate(runs: list):
    """
    Given a list of identically-structured dicts (one per seed), return
    (mean_dict, std_dict) with the same structure where each leaf scalar
    is replaced by its mean / std across seeds.
    """
    if not isinstance(runs[0], dict):
        valid = [v for v in runs if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not valid:
            return float("nan"), float("nan")
        return float(np.mean(valid)), float(np.std(valid))
    means, stds = {}, {}
    for k in runs[0].keys():
        vals = [r[k] for r in runs if k in r]
        if not vals:
            means[k] = float("nan")
            stds[k]  = float("nan")
        else:
            means[k], stds[k] = _recursive_aggregate(vals)
    return means, stds


def print_metrics_table_mean_std(mean_metrics: dict, std_metrics: dict, model_name: str = "Model"):
    """Like print_metrics_table but shows  mean ± std  at each cell."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {model_name}  (mean ± std over {len(EVAL_SEEDS)} seeds)")
    print(bar)

    def _fmt(m, s):
        if np.isnan(m):
            return "       nan"
        return f"{m:.3f}±{s:.3f}"

    print("\n[ Rollout metrics ]")
    header = f"{'metric':<30} {'overall':>14}" + "".join(f"  {n:>16}" for n in STYLE_NAMES.values())
    print(header)
    print("-" * len(header))
    for k in [
        "success_rate", "style_achievement_rate", "avg_return",
        "avg_episode_length", "detection_rate",
        "avg_enemy_distance", "avg_path_efficiency",
        "weapon_usage_rate", "camouflage_usage_rate",
    ]:
        ov = mean_metrics["rollout"]["overall"].get(k, float("nan"))
        os = std_metrics["rollout"]["overall"].get(k, float("nan"))
        row = f"{k:<30} {_fmt(ov, os):>14}"
        for sname in STYLE_NAMES.values():
            sv = mean_metrics["rollout"]["per_style"].get(sname, {}).get(k, float("nan"))
            ss = std_metrics["rollout"]["per_style"].get(sname, {}).get(k, float("nan"))
            row += f"  {_fmt(sv, ss):>16}"
        print(row)

    if mean_metrics.get("control_fidelity"):
        print("\n[ Control fidelity (mean ± std Spearman r / MSE) ]")
        cf_m, cf_s = mean_metrics["control_fidelity"], std_metrics["control_fidelity"]
        for dim_name in CONTROL_NAMES:
            r_m   = cf_m.get(dim_name, {}).get("spearman_r", float("nan"))
            r_s   = cf_s.get(dim_name, {}).get("spearman_r", float("nan"))
            mse_m = cf_m.get(dim_name, {}).get("mse",        float("nan"))
            mse_s = cf_s.get(dim_name, {}).get("mse",        float("nan"))
            print(f"  {dim_name:<22}  r = {_fmt(r_m, r_s)}  mse = {_fmt(mse_m, mse_s)}")
        mean_r   = cf_m.get("mean_abs_spearman_r", float("nan"))
        std_r    = cf_s.get("mean_abs_spearman_r", float("nan"))
        mean_mse = cf_m.get("mean_mse", float("nan"))
        std_mse  = cf_s.get("mean_mse", float("nan"))
        print(f"  {'mean |r|':<22}  {_fmt(mean_r, std_r)}")
        print(f"  {'mean mse':<22}  {_fmt(mean_mse, std_mse)}")

    if mean_metrics.get("latent"):
        print("\n[ Latent / reconstruction metrics ]")
        lat_m, lat_s = mean_metrics["latent"], std_metrics["latent"]
        for k in ["encoder_silhouette", "prior_silhouette", "encoder_style_accuracy",
                  "mean_kl_divergence", "action_accuracy"]:
            print(f"  {k:<30}  {_fmt(lat_m.get(k, float('nan')), lat_s.get(k, float('nan')))}")


# ---------------------------------------------------------------------------
# Main — load all models, run comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DEVICE = "cpu"
    HERE   = os.path.dirname(__file__)
    control_dim = 3

    dataset_params = {
        "sampling":                   True,
        "index_channel_only":         True,
        "state_normalization_factor":  1,
        "action_normalization_factor": 1,
        "max_len":                    20,
        "control_dim":                control_dim,
    }

    print("Loading dataset …")
    dataset = MiniGridDataset(trajectory_paths=paths, **dataset_params)

    # ------------------------------------------------------------------
    # Build adapters
    # ------------------------------------------------------------------
    adapters: Dict[str, ModelAdapter] = {}
    context_len = 8
    prompt_len = 2


    # 1. StyleVAE (main model)
    vae_model = StyleVAEPromptDT(
        state_dim=9, act_dim=7, hidden_size=128, latent_dim=16,
        max_length=20, max_ep_len=100, action_tanh=False,
        beta=0.0085, control_dim=control_dim, prior_hidden=128,
        free_bits=0.0, n_layer=4, n_head=8,
    )
    vae_ckpt = os.path.join(HERE, "trained_models/style_prompt_dt_minigrid_controls_condprior.pth")
    if os.path.exists(vae_ckpt):
        vae_model.load_state_dict(torch.load(vae_ckpt, map_location=DEVICE))
        print(f"Loaded StyleVAE: {vae_ckpt}")
    else:
        print(f"StyleVAE checkpoint not found at {vae_ckpt} — using random weights.")
    vae_model.to(DEVICE)
    adapters["StyleVAE"] = StyleVAEAdapter(vae_model, dataset, device=DEVICE)

    # 2. BC oracle (one policy per style)
    bc_policies: Dict[int, BCPolicy] = {}
    for sid in range(3):
        policy = BCPolicy(state_dim=9, act_dim=7, hidden_size=256, num_layers=3)
        ckpt = os.path.join(HERE, f"trained_models/bc_style{sid}.pth")
        if os.path.exists(ckpt):
            policy.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"Loaded BC style {sid}: {ckpt}")
        else:
            print(f"BC checkpoint not found: {ckpt} — using random weights.")
        policy.to(DEVICE)
        bc_policies[sid] = policy
    # BC was trained on raw (unnormalized) states — BCDataset uses normalize_state=False,
    # so state_mean=0 and state_std=1. Pass matching values here.
    _bc_state_dim = 9
    adapters["BC"] = BCAdapter(
        bc_policies,
        np.zeros(_bc_state_dim, dtype=np.float32),
        np.ones(_bc_state_dim,  dtype=np.float32),
        device=DEVICE,
    )

    # 3. PromptDT
    prompt_model = PromptingDecisionTransformer(
        state_dim=9, act_dim=7, hidden_size=128,
        max_length=context_len, max_ep_len=100, action_tanh=False,
        n_layer=4, n_head=8,
    )
    pdt_ckpt = os.path.join(HERE, "trained_models/prompt_dt_minigrid.pth")
    if os.path.exists(pdt_ckpt):
        prompt_model.load_state_dict(torch.load(pdt_ckpt, map_location=DEVICE))
        print(f"Loaded PromptDT: {pdt_ckpt}")
    else:
        print(f"PromptDT checkpoint not found: {pdt_ckpt} — using random weights.")
    prompt_model.to(DEVICE)
    adapters["PromptDT"] = PromptDTAdapter(prompt_model, dataset, prompt_length=prompt_len, device=DEVICE)


    # 4. SORL
    so_dataset = SODataset(
        trajectory_paths=paths,
        sampling=True,
        index_channel_only=True,
    )
    sorl_policies: Dict[int, SORLPolicy] = {}
    for k in range(3):
        p = SORLPolicy(state_dim=9, act_dim=7, hidden_size=256, num_layers=3)
        ckpt = os.path.join(HERE, f"trained_models/sorl_bc_style{k}.pth")
        if os.path.exists(ckpt):
            p.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"Loaded SORL style {k}: {ckpt}")
        else:
            print(f"SORL checkpoint not found: {ckpt} — using random weights.")
        p.to(DEVICE)
        sorl_policies[k] = p
    # style_map: adjust if print_cluster_composition() shows misaligned clusters
    adapters["SORL"] = SOAdapter(sorl_policies, so_dataset, device=DEVICE)

    # 5. ControlDT
    ctrl_model = ControlConditionedDT(
        state_dim=9, act_dim=7, hidden_size=128,
        control_dim=control_dim, max_length=context_len, max_ep_len=100,
        action_tanh=False, n_layer=4, n_head=8,
    )
    ctrl_ckpt = os.path.join(HERE, "trained_models/control_dt_minigrid.pth")
    if os.path.exists(ctrl_ckpt):
        ctrl_model.load_state_dict(torch.load(ctrl_ckpt, map_location=DEVICE))
        print(f"Loaded ControlDT: {ctrl_ckpt}")
    else:
        print(f"ControlDT checkpoint not found: {ctrl_ckpt} — using random weights.")
    ctrl_model.to(DEVICE)
    adapters["ControlDT"] = ControlDTAdapter(ctrl_model, dataset, device=DEVICE)

    # ------------------------------------------------------------------
    # Run evaluation over multiple seeds
    # ------------------------------------------------------------------
    eval_kwargs = dict(
        dataset=dataset,
        device=DEVICE,
        num_episodes_per_style=5,
        num_conditionings=5,
        max_ep_len=100,
        initial_rtg=1.0,
        max_context=context_len,
    )

    # per_seed_metrics[model_name] = list of result dicts, one per seed
    per_seed_metrics: Dict[str, list] = {name: [] for name in adapters}

    for seed in EVAL_SEEDS:
        print(f"\n{'#'*60}")
        print(f"  SEED {seed}")
        print(f"{'#'*60}")
        set_global_seeds(seed)
        for name, adapter in adapters.items():
            result = run_full_evaluation(adapter, **eval_kwargs, base_seed=seed)
            per_seed_metrics[name].append(result)

    # Aggregate across seeds
    mean_metrics: Dict[str, dict] = {}
    std_metrics:  Dict[str, dict] = {}
    for name, runs in per_seed_metrics.items():
        mean_metrics[name], std_metrics[name] = _recursive_aggregate(runs)

    # Print mean ± std tables
    for name in adapters:
        print_metrics_table_mean_std(mean_metrics[name], std_metrics[name], model_name=name)

    # Comparison table (mean only) + plots
    print_comparison_table(mean_metrics)
    plot_comparison(mean_metrics, save_dir=HERE, std_metrics=std_metrics)

import random
from typing import Callable
import numpy as np
import torch
import plotly.express as px
from functorch.einops import rearrange
from torch.utils.data import Dataset
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from collections import Counter
from dataset_utils.utils import TrajectoryReader


def controls_from_episode_summary(
    episode_summary,
    max_enemy_distance: float = 12.0,
) -> np.ndarray:
    """
    Build a float32 control vector from an episode_summary dict produced by
    MiniGridThreeStyles.  Returns zeros for any missing fields.

    Dimensions: [risk_tolerance, resource_pref, commitment]
    """
    if not isinstance(episode_summary, dict):
        episode_summary = {}

    min_dist        = float(episode_summary.get("min_enemy_distance", 0.0))
    avg_dist        = float(episode_summary.get("avg_enemy_distance", 0.0))
    path_efficiency = float(episode_summary.get("path_efficiency",    0.0))
    items_picked    = int(  episode_summary.get("items_picked",       0))
    picked_weapon      = float(bool(episode_summary.get("picked_weapon",      False)))
    picked_camouflage  = float(bool(episode_summary.get("picked_camouflage",  False)))

    norm_min = np.clip(min_dist / max_enemy_distance, 0.0, 1.0)
    norm_avg = np.clip(avg_dist / max_enemy_distance, 0.0, 1.0)

    risk_tolerance = 1.0 - norm_min
    resource_pref  = np.clip(items_picked / 2.0, 0.0, 1.0)
    stealth_pref   = np.clip(
        norm_avg * (1.0 - picked_weapon) + picked_camouflage * 0.9,
        0.0, 1.0,
    )
    safety_pref = np.clip(norm_avg + picked_camouflage * 0.3, 0.0, 1.0)
    commitment  = path_efficiency

    return np.array(
        [risk_tolerance, resource_pref, commitment],
        dtype=np.float32,
    )


# Marks an episode_summary as coming from MiniGridMultiStyles (the 4-style env),
# which logs per-step behavioural counters the 3-style env does not.
MULTI_STYLE_SUMMARY_KEYS = (
    "lava_steps",
    "enemy_adjacent_steps",
    "enemy_near_unprotected_steps",
    "detection_cone_steps",
)


def multi_style_controls_from_episode_summary(
    episode_summary,
    direct_dist: float = 14.0,
) -> np.ndarray:
    """
    Build a float32 control vector from an episode_summary dict produced by
    MiniGridMultiStyles (the 4-style env).  Returns zeros for missing fields.

    Dimensions: [risk_taking, stealth_exposure, confrontation]

    risk_taking      — how much danger it entered (worst of combat / lava /
                       half-weighted detection). low=bypass, high=daredevil.
    stealth_exposure — how hidden/safe it stayed; drops for UNPROTECTED exposure
                       only. high=bypass/camo, low=weapon/daredevil.
    confrontation    — how much it directly engaged the enemy (enemy-adjacent
                       steps). high=weapon, zero=bypass/daredevil.

    The normalising constants set the point at which a "full" engagement of each
    kind saturates near 1; they are tunable from the real count distributions
    after (re)collection — controls are recomputed on load, so retuning needs no
    re-collection.
    """
    if not isinstance(episode_summary, dict):
        episode_summary = {}

    total_steps = int(episode_summary.get("total_steps",                  0))
    lava        = int(episode_summary.get("lava_steps",                   0))
    adj_enemy   = int(episode_summary.get("enemy_adjacent_steps",         0))
    near_unprot = int(episode_summary.get("enemy_near_unprotected_steps", 0))
    cone        = int(episode_summary.get("detection_cone_steps",         0))

    ADJ_FULL, LAVA_FULL, CONE_FULL, EXPO_FULL = 3.0, 5.0, 4.0, 6.0

    adj_c  = np.clip(adj_enemy / ADJ_FULL,  0.0, 1.0)
    lava_c = np.clip(lava      / LAVA_FULL, 0.0, 1.0)
    cone_c = np.clip(cone      / CONE_FULL, 0.0, 1.0)
    expo_c = np.clip((near_unprot + lava) / EXPO_FULL, 0.0, 1.0)

    risk_taking      = np.clip(max(adj_c, lava_c, 0.5 * cone_c), 0.0, 1.0)
    stealth_exposure = np.clip(
        1.0 - 0.50 * adj_c - 0.80 * expo_c - 0.20 * cone_c, 0.0, 1.0
    )
    confrontation = adj_c

    return np.array(
        [risk_taking, stealth_exposure, confrontation],
        dtype=np.float32,
    )


# Marks an episode_summary as coming from MiniGridFourStyles.
FOUR_STYLE_SUMMARY_KEYS = ("risk_exposure_steps", "portal_steps")


def four_style_controls_from_episode_summary(
    episode_summary,
    risk_full: float = 6.0,
    stealth_full: float = 5.0,
) -> np.ndarray:
    """
    Build a float32 control vector from a MiniGridFourStyles episode_summary.

    Dimensions: [risk_taking, stealth_exposure, commitment]

    risk_taking      — steps spent ADJACENT to the active detection zone or lava
                       while carrying no protection (no camo/boots) and the enemy
                       is alive. Highest for portal (long x11 descent beside the
                       detection zone), then weapon; ~0 for camo (carries camo)
                       and daredevil (carries boots).
    stealth_exposure — immunity from danger: steps spent IN danger protected (in
                       the detection zone with camo, or on lava with boots). High
                       for camo and daredevil; ~0 for weapon and portal (they
                       never safely enter danger).
    commitment       — path directness (forward_steps / total_steps).

    Normalising constants saturate a "full" engagement near 1 and are tunable
    from the real count distributions; controls are recomputed on load.
    """
    if not isinstance(episode_summary, dict):
        episode_summary = {}

    # risk: prefer the dedicated counter (steps adjacent to the active detection
    # zone or lava, unprotected, enemy alive). For data collected before that
    # counter existed, fall back to near-enemy-unprotected steps + portal_steps,
    # where portal_steps stands in for "steps adjacent to the detection zone"
    # (that is what the portal descent is, in practice). lava_adjacent_steps is
    # intentionally NOT used: it isn't protection-aware and the only style that
    # walks beside lava (daredevil) carries boots, so it must stay low-risk.
    risk_exposure = episode_summary.get("risk_exposure_steps", None)
    if risk_exposure is None:
        risk_exposure = (int(episode_summary.get("enemy_near_unprotected_steps", 0))
                         + int(episode_summary.get("portal_steps", 0)))
    detection  = int(episode_summary.get("detection_steps", 0))
    lava       = int(episode_summary.get("lava_steps",      0))
    commitment = float(episode_summary.get("path_efficiency", 0.0))

    risk_taking      = np.clip(float(risk_exposure) / risk_full, 0.0, 1.0)
    stealth_exposure = np.clip((detection + lava) / stealth_full, 0.0, 1.0)

    return np.array(
        [risk_taking, stealth_exposure, commitment],
        dtype=np.float32,
    )


class TrajectoryDataset(Dataset):
    def __init__(
            self,
            trajectory_paths,
            sampling=False,
            index_channel_only=False,
            state_normalization_factor=1,
            action_normalization_factor=1,
            max_len=1,
            prob_go_from_end=0,
            pct_traj=1.0,
            rtg_scale=1,
            normalize_state=False,
            preprocess_observations: Callable = None,
            device="cpu",
            select_highest_count_return_group=False,
            highest_count_num_samples=500,
            highest_count_top_k=10,
    ):
        self.trajectory_paths = trajectory_paths
        self.max_len = max_len
        self.prob_go_from_end = prob_go_from_end
        self.pct_traj = pct_traj
        self.device = device
        self.normalize_state = normalize_state
        self.rtg_scale = rtg_scale
        self.preprocess_observations = preprocess_observations

        # used for input processing of the trajectory embedding model
        self.state_normalization_factor = state_normalization_factor
        self.action_normalization_factor = action_normalization_factor
        self.index_channel_only = index_channel_only
        self.sampling = sampling

        # TEMPORARY TEST: when True, ignore the length-based selection below and
        # keep only trajectories from the single most-common return group,
        # sampling `highest_count_num_samples` of them. See
        # get_highest_count_return_group.
        self.select_highest_count_return_group = select_highest_count_return_group
        self.highest_count_num_samples = highest_count_num_samples
        self.highest_count_top_k = highest_count_top_k

        self.load_trajectories()



    def load_trajectories(self) -> None:
        (merge_observations, merge_actions, merge_rewards, merge_returns, merge_dones,
         merge_truncated, merge_infos, merge_modes, merge_timesteps, merge_tasks) = [], [], [], [], [], [], [], [], [], []

        # used only for DEC-VAE experiments
        obs, acts, tasks = [], [], []

        # Iterating over many datasets with different environment modes or play
        # styles. Each entry in self.trajectory_paths may be either a single
        # path (str) or a group of paths (list/tuple). For a group we draw an
        # equal share from each sub-dataset and merge them under a single task
        # id, so e.g. a style's `_top` and `_bottom` runs become one combined,
        # half-and-half style. Single-path entries (e.g. daredevil) behave
        # exactly as before, so there is no need to duplicate them.
        for i, path_group in enumerate(self.trajectory_paths):
            sub_paths = [path_group] if isinstance(path_group, str) else list(path_group)
            n_sub = len(sub_paths)

            # total target per style/group, split as evenly as possible across
            # its sub-datasets (the first sub-path absorbs any remainder).
            group_samples = 1000
            per_sub = [group_samples // n_sub] * n_sub
            per_sub[0] += group_samples - sum(per_sub)

            for path, sub_samples in zip(sub_paths, per_sub):
                (states, actions, rewards, dones, truncated, returns,
                 timesteps, terminal_infos, data) = self._load_and_select(
                    path, sub_samples
                )

                # every sub-dataset in a group shares the same task id `i`
                tasks = np.ones(len(actions), dtype=np.int64) * i

                # merge datasets
                merge_actions.extend(actions)
                merge_rewards.extend(rewards)
                merge_dones.extend(dones)
                merge_truncated.extend(truncated)
                merge_observations.extend(states)
                merge_returns.extend(returns)
                merge_timesteps.extend(timesteps)
                merge_tasks.extend(tasks)
                merge_infos.extend(terminal_infos)
        self.actions = merge_actions
        self.rewards = merge_rewards
        self.dones = merge_dones
        self.truncated = merge_truncated
        self.states = merge_observations
        self.returns = merge_returns
        self.timesteps = merge_timesteps
        self.tasks = merge_tasks
        self.infos = merge_infos
        # ==================================
        # remove trajectories with length 0

        self.traj_lens = np.array([len(i) for i in self.states])
        traj_len_mask = self.traj_lens > 0
        self.actions = [i / self.action_normalization_factor for i, m in zip(self.actions, traj_len_mask) if m]
        self.rewards = [i for i, m in zip(self.rewards, traj_len_mask) if m]
        self.dones = [i for i, m in zip(self.dones, traj_len_mask) if m]
        self.truncated = [i for i, m in zip(self.truncated, traj_len_mask) if m]
        self.states = [i / self.state_normalization_factor for i, m in zip(self.states, traj_len_mask) if m]
        self.returns = [i for i, m in zip(self.returns, traj_len_mask) if m]
        self.timesteps = [i for i, m in zip(self.timesteps, traj_len_mask) if m]
        self.tasks = [i for i, m in zip(self.tasks, traj_len_mask) if m]
        self.infos = [i for i, m in zip(self.infos, traj_len_mask) if m]

        # Build per-trajectory control vectors from episode_summary.
        # MiniGridMultiStyles (4-style env) logs extra behavioural counters and
        # uses the new control set; the 3-style env keeps the original controls.
        # We pick per-trajectory by checking which fields the summary carries, so
        # both envs keep working without any change to the training scripts.
        def _controls_for(ep_info):
            summary = ep_info.get("episode_summary") if isinstance(ep_info, dict) else None
            if isinstance(summary, dict) and any(
                k in summary for k in FOUR_STYLE_SUMMARY_KEYS
            ):
                return four_style_controls_from_episode_summary(summary)
            if isinstance(summary, dict) and any(
                k in summary for k in MULTI_STYLE_SUMMARY_KEYS
            ):
                return multi_style_controls_from_episode_summary(summary)
            return controls_from_episode_summary(summary)

        self.controls = np.stack(
            [_controls_for(ep_info) for ep_info in self.infos],
            axis=0,
        )  # [N, control_dim]

        self.traj_lens = self.traj_lens[traj_len_mask]
        self.num_timesteps = sum(self.traj_lens)
        self.num_trajectories = len(self.states)

        self.state_dim = list(self.states[0][0].shape)
        self.act_dim = list(self.actions[0][0].shape)
        self.max_ep_len = max([len(i) for i in self.states])
        self.metadata = data["metadata"]

        self.indices = self.get_indices_of_top_p_trajectories(self.pct_traj)
        self.sampling_probabilities = self.get_sampling_probabilities()

        if self.normalize_state:
            self.state_mean, self.state_std = self.get_state_mean_std()
        else:
            self.state_mean = 0
            self.state_std = 1

        # TODO Make this way less hacky
        if self.preprocess_observations == one_hot_encode_observation:
            self.observation_type = "one_hot"

        # top_seq_lengths = self.get_top_trajectory_lengths(self.states, self.returns, top_k=6)
        # print(top_seq_lengths)
        # exit(0)



    def _load_and_select(self, path, num_samples):
        """Load a single trajectory file and return up to `num_samples`
        trajectories selected by the same top-length / return criteria used
        across the whole dataset.

        Returns per-trajectory lists (states, actions, rewards, dones,
        truncated, returns, timesteps, terminal_infos) plus the raw `data`
        dict (kept for metadata). Task ids are assigned by the caller so that
        several sub-datasets can share one id.
        """
        traj_reader = TrajectoryReader(path)
        data = traj_reader.read()
        observations = data["data"].get("observations")
        actions = data["data"].get("actions")
        rewards = data["data"].get("rewards")
        dones = data["data"].get("dones")
        truncated = data["data"].get("truncated")
        infos = data["data"].get("infos")

        observations = np.array(observations)
        T_steps, B_envs = observations.shape[0], observations.shape[1]
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # check whether observations are flat or an image
        if observations.shape[-1] == 3:
            # use state space that includes object IDX in each grid position
            if self.index_channel_only:
                observations = observations[:, :, :, :, 0]  # we use this in the VAE model
            else:
                observations = observations[:, :, :, :, :]  # We use this in the DT model
            self.observation_type = "index"
        elif observations.shape[-1] == 20:
            self.observation_type = "one_hot"
        else:
            raise ValueError(
                "Observations are not flat or images, check the shape of the observations: ",
                observations.shape,
            )
        if self.observation_type != "flat":
            if self.index_channel_only:
                t_observations = rearrange(
                    torch.tensor(observations), "t b h w  -> (b t) (h w)"
                )
            else:
                t_observations = rearrange(
                    torch.tensor(observations), "t b h w c -> (b t) h w c"
                )
        else:
            t_observations = rearrange(
                torch.tensor(observations), "t b f -> (b t) f"
            )

        t_actions = rearrange(torch.tensor(actions), "t b -> (b t)")
        t_rewards = rearrange(torch.tensor(rewards), "t b -> (b t)")
        t_dones = rearrange(torch.tensor(dones), "t b -> (b t)")
        t_truncated = rearrange(torch.tensor(truncated), "t b -> (b t)")
        t_done_or_truncated = torch.logical_or(t_dones, t_truncated)
        done_indices = torch.where(t_done_or_truncated)[0]

        # Extract the terminal info dict for each trajectory.
        # With gymnasium SyncVectorEnv, terminal infos live under info["final_info"][b].
        raw_terminal_infos = []
        if infos is not None:
            for flat_idx in done_indices.numpy():
                t_step = int(flat_idx) % T_steps
                b_env = int(flat_idx) // T_steps
                ep_info = {}
                step_info = infos[t_step]
                if isinstance(step_info, dict) and "final_info" in step_info:
                    fi = step_info["final_info"]
                    if fi is not None and b_env < len(fi) and fi[b_env] is not None:
                        ep_info = fi[b_env] if isinstance(fi[b_env], dict) else {}
                raw_terminal_infos.append(ep_info)
            raw_terminal_infos.append({})  # sentinel for the trailing empty segment
        else:
            raw_terminal_infos = [{} for _ in range(len(done_indices) + 1)]

        actions = torch.tensor_split(t_actions, done_indices + 1)
        rewards = torch.tensor_split(t_rewards, done_indices + 1)
        dones = torch.tensor_split(t_dones, done_indices + 1)
        truncated = torch.tensor_split(t_truncated, done_indices + 1)
        states = torch.tensor_split(t_observations, done_indices + 1)
        returns = [r.sum() for r in rewards]
        returns = ['%.2f' % elem for elem in returns]
        timesteps = [torch.arange(len(s)) for s in states]

        # Sampling trajectories based on their lengths and returns
        # top_seq_lengths = self.get_top_trajectory_lengths(states, returns, top_k=10)
        # print(top_seq_lengths)
        # seq_lens = [seq_len[0] for seq_len in top_seq_lengths]
        # top_seq_lengths = self.get_top_trajectory_returns(states, returns, top_k=20)
        # print([(len, ret, count) for (len, ret, count, idx) in top_seq_lengths])
        # print(len(states))
        top_seq_lengths = self.get_trajectories_above_return(states, returns, 1.59)
        print([(len, ret, count) for (len, ret, count, idx) in top_seq_lengths])
        index_lists = [idx for (_, _, _, index_len) in top_seq_lengths for idx in index_len]

        if self.select_highest_count_return_group:  # TEMPORARY TEST
            indexes = self.get_highest_count_return_group(
                states, returns, num_samples=self.highest_count_num_samples,
                top_k=self.highest_count_top_k)
        elif self.sampling:  # Use random sampled trajectories
            # index_lists = []
            # for seq_len in seq_lens:
            #     index_list = [index for index, (state, ret) in enumerate(zip(states, returns)) if
            #                   len(state) == seq_len]
            #     index_lists.extend(index_list)
            indexes = random.sample(index_lists, num_samples)
        else:  # Use non-random trajectories
            # indexes = [index for index, (state, ret) in enumerate(zip(states, returns)) if len(state) in seq_lens][-num_samples:]
            indexes = index_lists[-num_samples:]

        print(len(indexes))
        states = [states[idx] for idx in indexes]
        actions = [actions[idx] for idx in indexes]
        rewards = [rewards[idx] for idx in indexes]
        dones = [dones[idx] for idx in indexes]
        truncated = [truncated[idx] for idx in indexes]
        returns = [returns[idx] for idx in indexes]
        timesteps = [timesteps[idx] for idx in indexes]
        terminal_infos = [raw_terminal_infos[idx] for idx in indexes]

        # top_seq_lengths = self.get_top_trajectory_lengths(states, returns, top_k=20)
        # print(top_seq_lengths)
        # top_seq_lengths = self.get_top_trajectory_returns(states, returns, top_k=20)
        top_seq_lengths = self.get_trajectories_above_return(states, returns, 1.59)
        print(sum([count for (len, ret, count, idx) in top_seq_lengths]))


        return (states, actions, rewards, dones, truncated, returns,
                timesteps, terminal_infos, data)

    @staticmethod
    def get_top_trajectory_lengths(states, returns, top_k=5):
        lengths = [len(s) for s in states]
        returns = [float(r) for r in returns]
        freq = Counter(zip(lengths, returns))
        sorted_items = sorted(freq.items(), key=lambda x: (x[0][1], x[1]), reverse=True)
        top_seq_lengths = []
        seen_lengths = set()
        for (length, ret), count in sorted_items:
            if length not in seen_lengths:
                top_seq_lengths.append((length, ret, count))
                seen_lengths.add(length)
            if len(top_seq_lengths) == top_k:
                break
        return top_seq_lengths

    @staticmethod
    def get_top_trajectory_returns(states, returns, top_k=5):
        lengths = [len(s) for s in states]
        returns = [float(r) for r in returns]
        # group the sample indices by (length, ret) so we can recover them later
        groups = {}
        for idx, key in enumerate(zip(lengths, returns)):
            groups.setdefault(key, []).append(idx)
        sorted_items = sorted(groups.items(), key=lambda x: (x[0][1], len(x[1])), reverse=True)
        top_seq_returns = []
        seen_returns = set()
        for (length, ret), indices in sorted_items:
            if ret not in seen_returns:
                top_seq_returns.append((length, ret, len(indices), indices))
                seen_returns.add(ret)
            if len(top_seq_returns) == top_k:
                break
        return top_seq_returns

    @staticmethod
    def get_trajectories_above_return(states, returns, return_threshold):
        """Select every (length, return) group whose return is higher than
        `return_threshold`.

        Mirrors get_top_trajectory_lengths / get_top_trajectory_returns, but
        instead of keeping the top-k groups it keeps ALL groups with
        ret > return_threshold.

        Returns a list of (length, ret, count, indices) tuples, sorted by
        return desc (same shape as get_top_trajectory_returns).
        """
        lengths = [len(s) for s in states]
        returns = [float(r) for r in returns]
        groups = {}
        for idx, key in enumerate(zip(lengths, returns)):
            groups.setdefault(key, []).append(idx)
        seq_returns = [
            (length, ret, len(indices), indices)
            for (length, ret), indices in groups.items()
            if ret > return_threshold
        ]
        seq_returns.sort(key=lambda x: x[1], reverse=True)
        return seq_returns

    @staticmethod
    def get_highest_count_return_group(states, returns, num_samples=500, top_k=5):
        """TEMPORARY TEST: among the `top_k` highest-return groups, keep the one
        with the most trajectories.

        Groups trajectories by their return value, restricts to the `top_k`
        groups with the highest return, picks the most-common group among those,
        and randomly samples up to `num_samples` indices from it.

        Returns a list of selected trajectory indices.
        """
        returns = [float(r) for r in returns]
        groups = {}
        for idx, ret in enumerate(returns):
            groups.setdefault(ret, []).append(idx)

        # restrict to the top_k highest return values first ...
        top_returns = sorted(groups.keys(), reverse=True)[:top_k]
        # ... then pick the most-common group among those.
        best_ret = max(top_returns, key=lambda r: len(groups[r]))
        best_indices = groups[best_ret]
        n = min(num_samples, len(best_indices))
        print(
            f"[highest-count-return-group] top_k={top_k} "
            f"top_returns={sorted(top_returns, reverse=True)} -> chose return={best_ret} "
            f"count={len(best_indices)} sampling={n} "
            f"(out of {len(groups)} distinct return groups)"
        )
        return random.sample(best_indices, n)

    def get_indices_of_top_p_trajectories(self, pct_traj):
        num_timesteps = max(int(pct_traj * self.num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)

        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = self.num_trajectories - 1

        while (
                ind >= 0
                and timesteps + self.traj_lens[sorted_inds[ind]] < num_timesteps
        ):
            timesteps += self.traj_lens[sorted_inds[ind]]
            ind -= 1
            num_trajectories += 1

        sorted_inds = sorted_inds[-num_trajectories:]
        return sorted_inds

    def get_sampling_probabilities(self):
        p_sample = self.traj_lens[self.indices] / sum(
            self.traj_lens[self.indices]
        )
        return p_sample

    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for time in reversed(range(x.shape[0] - 1)):
            discount_cumsum[time] = x[time] + gamma * discount_cumsum[time + 1]
        return discount_cumsum

    def get_state_mean_std(self):
        # used for input normalization
        all_states = np.concatenate(self.states, axis=0)
        state_mean, state_std = (
            np.mean(all_states, axis=0),
            np.std(all_states, axis=0) + 1e-6,
        )
        return state_mean, state_std

    def get_batch(self, batch_size=256, max_len=100, prob_go_from_end=None):
        sorted_inds = self.indices

        batch_inds = np.random.choice(
            np.arange(len(sorted_inds)),
            size=batch_size,
            replace=True,
            p=self.sampling_probabilities,  # reweights so we sample according to timesteps
        )

        # initialize np arrays not lists
        states, actions, rewards, dones, rewards_to_gos, timesteps, mask = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(batch_size):
            # get the trajectory
            traj_index = sorted_inds[batch_inds[i]]

            s, a, r, d, rtg, ti, m = self.get_traj(
                traj_index, max_len, prob_go_from_end=prob_go_from_end
            )

            rewards.append(r)
            actions.append(a)
            states.append(s)
            dones.append(d)
            rewards_to_gos.append(rtg)
            mask.append(m)
            timesteps.append(ti)

        return self.return_tensors(states, actions, rewards, rewards_to_gos, dones, timesteps, mask)

    def get_traj(self, traj_index, max_len=100, prob_go_from_end=None):
        traj_rewards = self.rewards[traj_index]
        traj_states = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones = self.dones[traj_index]

        # TODO: configure this so non-sparse tasks are dealt with correctly!
        # This line is very slow if we use the "correct method"
        traj_rtg = np.ones(traj_rewards.shape) * traj_rewards[-1].item()

        # "Correct method"
        # traj_rtg = self.discount_cumsum(traj_rewards, gamma=1.0)

        # start index
        si = random.randint(0, traj_rewards.shape[0] - 1)
        if prob_go_from_end is not None:
            if random.random() < prob_go_from_end:
                si = traj_rewards.shape[0] - max_len
                si = max(0, si)  # make sure it's not negative

        # get sequences from dataset
        s = traj_states[si: si + max_len].reshape(1, -1, *self.state_dim)
        a = traj_actions[si: si + max_len].reshape(1, -1, *self.act_dim)
        r = traj_rewards[si: si + max_len].reshape(1, -1, 1)
        rtg = traj_rtg[si: si + max_len].reshape(1, -1, 1)
        d = traj_dones[si: si + max_len].reshape(1, -1)
        ti = np.arange(si, si + s.shape[1]).reshape(1, -1)

        # sometime the trajectory is shorter than max_len (due to random start index or end of episode)
        tlen = s.shape[1]

        # sanity check
        assert tlen <= max_len, f"tlen: {tlen} max_len: {max_len}"

        padding_required = max_len - tlen
        s = self.add_padding(s, 0, padding_required)
        a = self.add_padding(a, -10, padding_required)
        r = self.add_padding(r, 0, padding_required)
        rtg = self.add_padding(rtg, rtg[0, -1], padding_required)
        d = self.add_padding(d, 2, padding_required)
        ti = self.add_padding(ti, 0, padding_required)
        m = self.add_padding(np.ones((1, tlen)), 0, padding_required)

        # padding and state + reward normalization
        s = (s - self.state_mean) / self.state_std
        rtg = rtg / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    def add_padding(self, tokens, padding_token, padding_required):
        if padding_required > 0:
            return np.concatenate(
                [
                    np.ones((1, padding_required, *tokens.shape[2:]))
                    * padding_token,
                    tokens,
                ],
                axis=1,
            )
        return tokens

    def return_tensors(self, s, a, r, rtg, d, timesteps, mask):
        if isinstance(s, torch.Tensor):
            s = s.to(dtype=torch.float32, device=self.device)
        else:
            s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)

        if isinstance(a, torch.Tensor):
            a = a.to(dtype=torch.long, device=self.device)
        else:
            a = torch.from_numpy(a).to(dtype=torch.long, device=self.device)

        if isinstance(r, torch.Tensor):
            r = r.to(dtype=torch.float32, device=self.device)
        else:
            r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)

        if isinstance(rtg, torch.Tensor):
            rtg = rtg.to(dtype=torch.float32, device=self.device)
        else:
            rtg = torch.from_numpy(rtg).to(
                dtype=torch.float32, device=self.device
            )

        if isinstance(d, torch.Tensor):
            d = d.to(dtype=torch.bool, device=self.device)
        else:
            d = torch.from_numpy(d).to(dtype=torch.bool, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(
            dtype=torch.long, device=self.device
        )
        mask = torch.from_numpy(mask).to(dtype=torch.bool, device=self.device)

        # squeeze out the batch dimension
        s = s.squeeze(0)
        a = a.squeeze(0)
        r = r.squeeze(0)
        rtg = rtg.squeeze(0)
        d = d.squeeze(0)
        timesteps = timesteps.squeeze(0)
        mask = mask.squeeze(0)

        # TODO fix the order of d, rtg here.
        return s, a, r, d, rtg, timesteps, mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_index = self.indices[idx]

        s, a, r, d, rtg, ti, m = self.get_traj(traj_index, max_len=self.max_len,
                                                             prob_go_from_end=self.prob_go_from_end)
        if self.preprocess_observations is not None:
            s = self.preprocess_observations(s)
        return s, a, r, d, rtg, ti, m


class TrajectoryVisualizer:
    def __init__(self, trajectory_dataset: TrajectoryDataset):
        self.trajectory_loader = trajectory_dataset

    def plot_reward_over_time(self):
        reward = [i[-1] for i in self.trajectory_loader.rewards if len(i) > 0]
        timesteps = [
            i.max() for i in self.trajectory_loader.timesteps if len(i) > 0
        ]

        # create a categorical color array for reward <0, 0, >0
        colors = np.zeros(len(reward))
        colors[np.array(reward) < 0] = -1
        colors[np.array(reward) > 0] = 1

        color_map = {-1: "Negative", 0: "Zero", 1: "Positive"}

        fig = px.scatter(
            y=reward,
            x=timesteps,
            color=[color_map[i] for i in colors],
            title="Reward vs Timesteps",
            template="plotly_white",
            labels={
                "x": "Timesteps",
                "y": "Reward",
            },
            marginal_x="histogram",
            marginal_y="histogram",
        )

        return fig

    def plot_base_action_frequencies(self):
        fig = px.bar(
            y=torch.concat(self.trajectory_loader.actions).bincount()
            # x=[IDX_TO_ACTION[i] for i in range(7)],
            # color=[IDX_TO_ACTION[i] for i in range(7)],
        )

        fig.update_layout(
            title="Base Action Frequencies",
            xaxis_title="Action",
            yaxis_title="Frequency",
        )

        return fig


def one_hot_encode_observation(img: torch.Tensor) -> torch.Tensor:
    """Converts a batch of observations into one-hot encoded numpy arrays."""

    img = img.to(int)  # .numpy()
    batch_size, height, width, num_channels = img.shape
    num_bits = 20
    new_observation_space = (batch_size, height, width, num_bits)

    out = np.zeros(new_observation_space, dtype="uint8")

    for b in range(batch_size):
        for i in range(height):
            for j in range(width):
                value = img[b, i, j, 0]
                color = img[b, i, j, 1]
                state = img[b, i, j, 2]

                out[b, i, j, value] = 1
                out[b, i, j, len(OBJECT_TO_IDX) + color] = 1
                out[
                    b, i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state
                ] = 1

    return torch.from_numpy(out).float()


if __name__ == '__main__':
    paths = [
        # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-0.gz",
        # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-1.gz",
        # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_goal2.gz",
        # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_goal3.gz",

        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_bypass.gz",
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_weapon.gz",
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_backstab.gz",

    ]
    trajectory_data_set = TrajectoryDataset(trajectory_paths=paths, sampling=True)
    print(trajectory_data_set.states[-1].shape)

import abc
import json
import os
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch as t
from torch import nn, optim
from torch.distributions.categorical import Categorical

from new_implementation.configs import (
    EnvironmentConfig,
    LSTMModelConfig,
    OnlineTrainConfig,
    RunConfig,
    TransformerModelConfig,
)
from new_implementation.env import make_env

from loss_functions import (
    calc_clipped_surrogate_objective,
    calc_entropy_bonus,
    calc_value_function_loss,
)
from memory import Memory, process_memory_vars_to_log
from sampling_methods import sample_from_categorical
from trajectory_writer import TrajectoryWriter
from util import get_obs_shape


class PPOScheduler:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lr: float,
        end_lr: float,
        num_updates: int,
    ):
        """
        A learning rate scheduler for a Proximal Policy Optimization (PPO) algorithm.

        Args:
        - optimizer (optim.Optimizer): the optimizer to use for updating the learning rate.
        - initial_lr (float): the initial learning rate.
        - end_lr (float): the end learning rate.
        - num_updates (int): the number of updates to perform before the learning rate reaches end_lr.
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        """
        Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.
        """
        self.n_step_calls += 1
        frac = self.n_step_calls / self.num_updates
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (
                self.end_lr - self.initial_lr
            )


class PPOAgent(nn.Module):
    critic: nn.Module
    actor: nn.Module

    @abc.abstractmethod
    def __init__(self, envs: gym.vector.SyncVectorEnv, device):
        super().__init__()
        self.envs = envs
        self.device = device

        self.critic = nn.Sequential()
        self.actor = nn.Sequential()

    def make_optimizer(
        self, num_updates: int, initial_lr: float, end_lr: float
    ) -> Tuple[optim.Optimizer, PPOScheduler]:
        """Returns an Adam optimizer with a learning rate schedule for updating the agent's parameters.

        Args:
            num_updates (int): The total number of updates to be performed.
            initial_lr (float): The initial learning rate.
            end_lr (float): The final learning rate.

        Returns:
            Tuple[optim.Optimizer, PPOScheduler]: A tuple containing the optimizer and its attached scheduler.
        """
        optimizer = optim.Adam(
            self.parameters(), lr=initial_lr, eps=1e-5, maximize=True
        )
        scheduler = PPOScheduler(optimizer, initial_lr, end_lr, num_updates)
        return (optimizer, scheduler)

    @abc.abstractmethod
    def rollout(self, memory, args, envs, trajectory_writer, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def learn(self, memory, args, optimizer, scheduler) -> None:
        pass

    def layer_init(
        self,
        layer: nn.Linear,
        std: float = np.sqrt(2),
        bias_const: float = 0.0,
    ) -> nn.Linear:
        """Initializes the weights of a linear layer with orthogonal
        initialization and the biases with a constant value.

        Args:
            layer (nn.Linear): The linear layer to be initialized.
            std (float, optional): The standard deviation of the
                distribution used to initialize the weights. Defaults to np.sqrt(2).
            bias_const (float, optional): The constant value to initialize the biases with. Defaults to 0.0.

        Returns:
            nn.Linear: The initialized linear layer.
        """
        t.nn.init.orthogonal_(layer.weight, std)
        t.nn.init.constant_(layer.bias, bias_const)
        return layer


class FCAgent(PPOAgent):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        environment_config: EnvironmentConfig,
        fc_model_config=None,  # not necessary yet but keeps type signatures the same
        device: t.device = t.device("cpu"),
        hidden_dim: int = 64,
    ):
        """
        An agent for a Proximal Policy Optimization (PPO) algorithm.

        Args:
        - envs (gym.vector.SyncVectorEnv): the environment(s) to interact with.
        - device (t.device): the device on which to run the agent.
        - hidden_dim (int): the number of neurons in the hidden layer.
        """
        super().__init__(envs=envs, device=device)

        self.environment_config = environment_config
        self.model_config = (
            fc_model_config
            if fc_model_config is not None
            else {"hidden_dim": hidden_dim}
        )
        self.obs_shape = get_obs_shape(envs.single_observation_space)
        self.num_obs = np.array(self.obs_shape).prod()
        self.num_actions = envs.single_action_space.n
        self.hidden_dim = (
            hidden_dim if hidden_dim else fc_model_config["hidden_dim"]
        )
        self.critic = nn.Sequential(
            nn.Flatten(),
            self.layer_init(nn.Linear(self.num_obs, self.hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            nn.Flatten(),
            self.layer_init(nn.Linear(self.num_obs, self.hidden_dim)),
            nn.Tanh(),
            self.layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            self.layer_init(
                nn.Linear(self.hidden_dim, self.num_actions), std=0.01
            ),
        )
        self.device = device
        self = self.to(device)

    def rollout(
        self,
        memory: Memory,
        num_steps: int,
        envs: gym.vector.SyncVectorEnv,
        trajectory_writer=None,
        sampling_method="basic",
        **kwargs,
    ) -> None:
        """Performs the rollout phase of the PPO algorithm, collecting experience by interacting with the environment.

        Args:
            memory (Memory): The replay buffer to store the experiences.
            num_steps (int): The number of steps to collect.
            envs (gym.vector.SyncVectorEnv): The vectorized environment to interact with.
            trajectory_writer (TrajectoryWriter, optional): The writer to log the
                collected trajectories. Defaults to None.
        """

        device = memory.device
        cuda = device == "cuda"
        obs = memory.next_obs
        done = memory.next_done

        for _ in range(num_steps):
            with t.inference_mode():
                logits = self.actor(obs)
                value = self.critic(obs).flatten()
            probs = Categorical(logits=logits)
            action = sample_from_categorical(probs, sampling_method, **kwargs)
            logprob = probs.log_prob(action)
            next_obs, reward, next_done, next_truncated, info = envs.step(
                action.cpu().numpy()
            )
            next_obs = memory.obs_preprocessor(next_obs)
            reward = t.from_numpy(reward).to(device)

            if trajectory_writer is not None:
                obs_np = (
                    obs.detach().cpu().numpy()
                    if cuda
                    else obs.detach().numpy()
                )
                reward_np = (
                    reward.detach().cpu().numpy()
                    if cuda
                    else reward.detach().numpy()
                )
                action_np = (
                    action.detach().cpu().numpy()
                    if cuda
                    else action.detach().numpy()
                )
                trajectory_writer.accumulate_trajectory(
                    next_obs=obs_np,
                    reward=reward_np,
                    action=action_np,
                    done=next_done,
                    truncated=next_truncated,
                    info=info,
                )
            # Store (s_t, d_t, a_t, logpi(a_t|s_t), v(s_t), r_t+1)
            memory.add(info, obs, done, action, logprob, value, reward)
            obs = t.from_numpy(next_obs).to(device)
            done = t.from_numpy(next_done).to(device, dtype=t.float)

        # Store last (obs, done, value) tuple, since we need it to compute advantages
        memory.next_obs = obs
        memory.next_done = done
        with t.inference_mode():
            memory.next_value = self.critic(obs).flatten()

    def learn(
        self,
        memory: Memory,
        args: OnlineTrainConfig,
        optimizer: optim.Optimizer,
        scheduler: PPOScheduler,
        track: bool,
    ) -> None:
        """Performs the learning phase of the PPO algorithm, updating the agent's parameters
        using the collected experience.

        Args:
            memory (Memory): The replay buffer containing the collected experiences.
            args (OnlineTrainConfig): The configuration for the training.
            optimizer (optim.Optimizer): The optimizer to update the agent's parameters.
            scheduler (PPOScheduler): The scheduler attached to the optimizer.
            track (bool): Whether to track the training progress.
        """
        for _ in range(args.update_epochs):
            minibatches = memory.get_minibatches()
            # Compute loss on each minibatch, and step the optimizer
            for mb in minibatches:
                logits = self.actor(mb.obs)
                probs = Categorical(logits=logits)
                values = self.critic(mb.obs).squeeze()
                clipped_surrogate_objective = calc_clipped_surrogate_objective(
                    probs,
                    mb.actions,
                    mb.advantages,
                    mb.logprobs,
                    args.clip_coef,
                )
                value_loss = calc_value_function_loss(
                    values, mb.returns, args.vf_coef
                )
                entropy_bonus = calc_entropy_bonus(probs, args.ent_coef)
                total_objective_function = (
                    clipped_surrogate_objective - value_loss + entropy_bonus
                )
                optimizer.zero_grad()
                total_objective_function.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Get debug variables, for just the most recent minibatch (otherwise there's too much logging!)
        if track:
            with t.inference_mode():
                newlogprob = probs.log_prob(mb.actions)
                logratio = newlogprob - mb.logprobs
                ratio = logratio.exp()
                approx_kl = (ratio - 1 - logratio).mean().item()
                clipfracs = [
                    ((ratio - 1.0).abs() > args.clip_coef)
                    .float()
                    .mean()
                    .item()
                ]
            memory.add_vars_to_log(
                learning_rate=optimizer.param_groups[0]["lr"],
                avg_value=values.mean().item(),
                value_loss=value_loss.item(),
                clipped_surrogate_objective=clipped_surrogate_objective.item(),
                entropy=entropy_bonus.item(),
                approx_kl=approx_kl,
                clipfrac=np.mean(clipfracs),
            )


def get_agent(
    envs: gym.vector.SyncVectorEnv,
    environment_config: EnvironmentConfig,
    online_config,
) -> PPOAgent:
    """
    Returns an agent based on the given configuration.

    Args:
    - model_config: The configuration for the transformer model.
    - envs: The environment to train on.
    - environment_config: The configuration for the environment.
    - online_config: The configuration for online training.

    Returns:
    - An agent.
    """
    agent = FCAgent(
        envs,
        environment_config=environment_config,
        device=environment_config.device,
        hidden_dim=online_config.hidden_size,
    )
    return agent


def load_saved_checkpoint(path, num_envs=10) -> PPOAgent:
    # load the config from the checkpoint
    saved_state = t.load(path, map_location=t.device("cpu"))
    # assert all the fields we need are present
    assert (
        "environment_config" in saved_state
    ), "environment_config not found in checkpoint"
    assert (
        "model_config" in saved_state
    ), "model_config not found in checkpoint"
    assert (
        "model_state_dict" in saved_state
    ), "model_state_dict not found in checkpoint"
    assert (
        "online_config" in saved_state
    ), "online_config not found in checkpoint"

    # create the environment
    environment_config = EnvironmentConfig(
        **json.loads(saved_state["environment_config"])
    )
    envs = gym.vector.SyncVectorEnv(
        [make_env(environment_config, 0, 0, "test", mode=environment_config.env_mode) for _ in range(num_envs)]
    )

    # create the model config
    other_args = json.loads(saved_state["model_config"])
    # remove environment config from the model config
    if "environment_config" in other_args:
        del other_args["environment_config"]

    # get the online config
    online_config_args = json.loads(saved_state["online_config"])
    # remove batch_size and minibatch_size from the online config
    if "batch_size" in online_config_args:
        del online_config_args["batch_size"]
    if "minibatch_size" in online_config_args:
        del online_config_args["minibatch_size"]
    online_config = OnlineTrainConfig(**online_config_args)

    # TODO: this is a hack, fix it
    model_config = None
    if "n_ctx" in other_args:
        model_config = TransformerModelConfig(**other_args)
    elif "use_memory" in other_args:
        model_config = LSTMModelConfig(environment_config, **other_args)

    # create the model
    agent = get_agent(
        model_config=model_config,
        envs=envs,
        environment_config=environment_config,
        online_config=online_config,
    )

    # load the model state from the checkpoint
    agent.load_state_dict(saved_state["model_state_dict"])

    # return the model
    return agent


def load_all_agents_from_checkpoints(checkpoint_folder_path, num_envs=10):
    """
        Example:
    --------
    .. code-block:: python
        >>>  import wandb
        >>>  run = wandb.init()
        >>>  artifact = run.use_artifact('arena-ldn/PPO-MiniGrid/Test-PPO-LSTM_checkpoints:v16', type='model')
        >>>  artifact_dir = artifact.download()

        >>>  checkpoint_folder_path = "artifacts/Test-PPO-LSTM_checkpoints:v16"
        >>>  agents = load_all_agents_from_checkpoints(checkpoint_folder_path)

    """
    # Get all files in the checkpoint folder
    checkpoint_files = os.listdir(checkpoint_folder_path)

    # Filter out non-checkpoint files
    checkpoint_files = [f for f in checkpoint_files if f.endswith(".pt")]

    # Load each checkpoint into an agent
    agents = []
    for checkpoint_file in checkpoint_files:
        path = os.path.join(checkpoint_folder_path, checkpoint_file)
        agent = load_saved_checkpoint(path, num_envs=num_envs)
        agents.append(agent)

    return agents


def sample_from_agents(
    agents,
    rollout_length=2000,
    trajectory_path=None,
    num_envs=1,
    sampling_method="basic",
):
    all_episode_lengths = []
    all_episode_returns = []

    # Sample rollouts from each agent
    for i, agent in enumerate(agents):
        memory = Memory(
            agent.envs,
            OnlineTrainConfig(num_envs=num_envs),
            device=agent.device,
        )
        if trajectory_path:
            trajectory_writer = TrajectoryWriter(
                path=os.path.join(trajectory_path, f"rollouts_agent_{i}.gz"),
                run_config=RunConfig(track=False),
                environment_config=agent.environment_config,
                online_config=OnlineTrainConfig(num_envs=num_envs),
                model_config=agent.model_config,
            )
        else:
            trajectory_writer = None
        agent.rollout(
            memory,
            rollout_length,
            agent.envs,
            trajectory_writer,
            sampling_method,
        )
        if trajectory_writer:
            trajectory_writer.tag_terminated_trajectories()
            trajectory_writer.write(upload_to_wandb=False)

        # Process the episode lengths and returns
        df = process_memory_vars_to_log(memory.vars_to_log)
        all_episode_lengths.append(df["episode_length"])
        all_episode_returns.append(df["episode_return"])

    return all_episode_lengths, all_episode_returns

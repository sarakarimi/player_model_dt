import torch as t
import torch.nn as nn
from dataclasses import asdict

from functorch.einops import rearrange
from torch.utils.data import random_split, WeightedRandomSampler, DataLoader
from tqdm import tqdm

import wandb
from configs import EnvironmentConfig, OfflineTrainConfig
from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset
from eval import evaluate_dt_agent
from util import configure_optimizers, get_scheduler
from style_decision_transformer import (
    TrajectoryTransformer,
)


def train(
        model: TrajectoryTransformer,
        trajectory_data_set: TrajectoryDataset,
        make_env,
        offline_config: OfflineTrainConfig,
        device="cpu",
):
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)

    train_dataloader, test_dataloader = get_dataloaders(
        trajectory_data_set, offline_config
    )

    # get optimizer from string
    optimizer = configure_optimizers(model, offline_config)
    # TODO: Stop passing through all the args to the scheduler, shouldn't be necessary.
    scheduler_config = asdict(offline_config)
    del scheduler_config["optimizer"]
    # get total number of training steps.
    train_batches_per_epoch = len(train_dataloader)
    scheduler_config["training_steps"] = (
            offline_config.train_epochs * train_batches_per_epoch
    )
    scheduler = get_scheduler(
        offline_config.scheduler, optimizer, **scheduler_config
    )
    # can uncomment this to get logs of gradients and pars.
    # wandb.watch(model, log="all", log_freq=train_batches_per_epoch)
    prompt = None
    pbar = tqdm(range(offline_config.train_epochs))
    for epoch in pbar:
        for batch, traj in enumerate(train_dataloader):
            if offline_config.soft_prompt_mode is not None:
                s, a, r, d, rtg, ti, m, prompt = traj
            else:
                s, a, r, d, rtg, ti, m = traj



            total_batches = epoch * train_batches_per_epoch + batch

            model.train()

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)
            action_space = model.environment_config.action_space.n
            a[a == -10] = action_space  # dummy action for padding

            optimizer.zero_grad()

            action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None
            _, action_preds, _ = model(
                states=s,
                # remove last action
                actions=action,
                rtgs=rtg,  # remove last rtg
                timesteps=ti.unsqueeze(-1),
                prompt=prompt
            )
            action_preds = rearrange(action_preds, "b t a -> (b t) a")
            a_exp = rearrange(a, "b t -> (b t)").to(t.int64)

            # ignore dummy action
            loss = loss_fn(
                action_preds[a_exp != action_space],
                a_exp[a_exp != action_space],
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Training DT: {loss.item():.4f}")

            if offline_config.track:
                # tokens_seen = (
                #     (total_batches + 1)
                #     * offline_config.batch_size
                #     * model.transformer_config.n_ctx
                # )
                # learning_rate = optimizer.param_groups[0]["lr"]
                wandb.log({"train/loss": loss.item()}, step=total_batches)
                # wandb.log(
                #     {"metrics/tokens_seen": tokens_seen}, step=total_batches
                # )
                # wandb.log(
                #     {"metrics/learning_rate": learning_rate},
                #     step=total_batches,
                # )

        # # at test frequency
        if epoch % offline_config.test_frequency == 0:
            test(
                model=model,
                dataloader=test_dataloader,
                epochs=offline_config.test_epochs,
                track=offline_config.track,
                batch_number=total_batches,
                soft_prompt_mode=offline_config.soft_prompt_mode
            )

        if epoch % offline_config.eval_frequency == 0:
            eval_env_config = EnvironmentConfig(
                capture_video=True,
                max_steps=min(
                    model.environment_config.max_steps,
                    offline_config.eval_max_time_steps,
                ),
                fully_observed=False,
                one_hot_obs=(trajectory_data_set.observation_type == "one_hot"),
                view_size=model.environment_config.observation_space["image"].shape[0]
                if "image" in list(model.environment_config.observation_space.keys())
                else 7,
            )

            prompt = None
            for rtg in offline_config.initial_rtg:
                for i, style_id in enumerate(eval_env_config.env_styles):
                    if offline_config.soft_prompt_mode is not None:
                        prompt = trajectory_data_set.sample_random_prompts(task_label=i, num_samples=offline_config.eval_num_envs)

                    eval_env_func = make_env(
                        config=eval_env_config,
                        seed=batch,
                        idx=0,
                        run_name=f"dt_eval_videos_{batch}",
                        mode=style_id
                    )
                    evaluate_dt_agent(
                        model=model,
                        env_func=eval_env_func,
                        trajectories=offline_config.eval_episodes,
                        track=offline_config.track,
                        batch_number=total_batches,
                        initial_rtg=float(rtg),
                        device=device,
                        num_envs=offline_config.eval_num_envs,
                        soft_prompt_mode=offline_config.soft_prompt_mode,
                        prompt=prompt,
                        style_id=style_id
                    )

    return model


def test(
        model: TrajectoryTransformer,
        dataloader: DataLoader,
        epochs=10,
        track=False,
        batch_number=0,
        soft_prompt_mode=False,
):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss = 0
    n_correct = 0
    n_actions = 0

    pbar = tqdm(range(epochs))
    test_batches_per_epoch = len(dataloader)

    prompt = None
    for epoch in pbar:
        for batch, traj in enumerate(dataloader):
            if soft_prompt_mode is not None:
                s, a, r, d, rtg, ti, m, prompt = traj
            else:
                s, a, r, d, rtg, ti, m = traj

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)
            action_space = model.environment_config.action_space.n
            a[a == -10] = action_space

            _, action_preds, _ = model(
                states=s,
                actions=a[:, :-1].unsqueeze(-1)
                if a.shape[1] > 1
                else None,
                rtgs=rtg,
                timesteps=ti.unsqueeze(-1),
                prompt=prompt,
            )

            action_preds = rearrange(action_preds, "b t a -> (b t) a")
            a_exp = rearrange(a, "b t -> (b t)").to(t.int64)

            a_hat = t.argmax(action_preds, dim=-1)
            a_exp = rearrange(a, "b t -> (b t)").to(t.int64)

            action_preds = action_preds[a_exp != action_space]
            a_hat = a_hat[a_exp != action_space]
            a_exp = a_exp[a_exp != action_space]

            n_actions += a_exp.shape[0]
            n_correct += (a_hat == a_exp).sum()
            loss += loss_fn(action_preds, a_exp)

            accuracy = n_correct.item() / n_actions
            pbar.set_description(f"Testing DT: Accuracy so far {accuracy:.4f}")

    mean_loss = loss.item() / (epochs * test_batches_per_epoch)

    if track:
        wandb.log({"test/loss": mean_loss}, step=batch_number)
        wandb.log({"test/accuracy": accuracy}, step=batch_number)

    return mean_loss, accuracy


def get_dataloaders(trajectory_data_set, offline_config):
    train_dataset, test_dataset = random_split(
        trajectory_data_set, [0.90, 0.10]
    )

    # Create the train DataLoader
    train_sampler = WeightedRandomSampler(
        weights=trajectory_data_set.sampling_probabilities[
            train_dataset.indices
        ],
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=offline_config.batch_size,
        sampler=train_sampler,
    )

    # Create the test DataLoader
    test_sampler = WeightedRandomSampler(
        weights=trajectory_data_set.sampling_probabilities[
            test_dataset.indices
        ],
        num_samples=len(test_dataset),
        replacement=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=offline_config.batch_size,
        sampler=test_sampler,
    )

    return train_dataloader, test_dataloader

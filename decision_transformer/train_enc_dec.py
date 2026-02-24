import torch as t
import torch.nn as nn
from dataclasses import asdict

from functorch.einops import rearrange
from torch.utils.data import random_split, WeightedRandomSampler, DataLoader
from tqdm import tqdm
import numpy as np

import wandb
from configs import EnvironmentConfig, OfflineTrainConfig
from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset
from decision_transformer.style_enc_decision_transformer_dec import VariationalStyleDecisionTransformer
from eval_enc_dec import evaluate_dt_agent
from util import configure_optimizers, get_scheduler
from style_decision_transformer import (
    TrajectoryTransformer,
)
from trajectory_embedding.style_dec_vae.lstm.style_vae import cluster_latents, plot_embeddings


def train_enc_dec(
        model: VariationalStyleDecisionTransformer,
        trajectory_data_set: TrajectoryDataset,
        make_env,
        offline_config: OfflineTrainConfig,
        device="cpu",
):
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)

    train_dataloader, test_dataloader = get_dataloaders(
        trajectory_data_set, offline_config, collate_fn=trajectory_data_set.collate_fn
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
        # Lists to collect z and labels for plotting
        Z = []
        labels_list = []
        should_plot = (epoch % offline_config.test_frequency == 0)

        for batch, batch_dict in enumerate(train_dataloader):
            # Unpack batch dictionary
            s = batch_dict["states"]
            a = batch_dict["actions"]
            r = batch_dict["rewards"]
            d = batch_dict["dones"]
            rtg = batch_dict["returns_to_go"]
            ti = batch_dict["timesteps"]
            m = batch_dict["attention_mask"]

            full_states = batch_dict["full_states"]
            full_actions = batch_dict["full_actions"]
            full_timesteps = batch_dict["full_timesteps"]
            full_attn_mask = batch_dict["full_attention_mask"]
            task_labels = batch_dict["task_labels"]

            total_batches = epoch * train_batches_per_epoch + batch

            model.train()

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)
            action_space = model.environment_config.action_space.n
            a[a == -10] = action_space  # dummy action for padding

            optimizer.zero_grad()

            action = a[:, :-1].unsqueeze(-1) if a.shape[1] > 1 else None
            output = model(
                states=s,
                # remove last action
                actions=action,
                rtgs=rtg,
                timesteps=ti.unsqueeze(-1),
                full_states=full_states,
                full_actions=full_actions,
                full_timesteps=full_timesteps,
                full_attn_masks=full_attn_mask,
            )
            action_preds = output['action_preds']
            action_preds = rearrange(action_preds, "b t a -> (b t) a")
            a_exp = rearrange(a, "b t -> (b t)").to(t.int64)

            # Collect z and labels for plotting if this is a test epoch
            if should_plot:
                Z.append(output['z'].detach().cpu().numpy())
                labels_list.append(task_labels.detach().cpu().numpy())

            # ignore dummy action
            dt_loss = loss_fn(
                action_preds[a_exp != action_space],
                a_exp[a_exp != action_space],
            )
            kl_loss = model.compute_kl_loss(output["mu"], output["logvar"])
            loss = dt_loss + kl_loss
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
            )

            # Plot embeddings from training data collected during this epoch
            true_labels = np.concatenate(labels_list, 0)
            Z_concat = np.concatenate(Z, 0)
            n_clusters = len(np.unique(true_labels))
            predicted_labels, cluster_centroids = cluster_latents(Z_concat, n_clusters)
            plot_embeddings(gtruth=predicted_labels, Z=Z_concat, label_name='task_predicted')
            plot_embeddings(gtruth=true_labels, Z=Z_concat, label_name='task_ground_truth')

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

            for rtg in offline_config.initial_rtg:
                for i, style_id in enumerate(eval_env_config.env_styles):
                    # Sample full trajectories for encoder
                    full_states, full_actions, full_timesteps, full_attention_mask = (
                        trajectory_data_set.sample_random_prompts(
                            task_label=i,
                            num_samples=offline_config.eval_num_envs
                        )
                    )

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
                        full_states=full_states,
                        full_actions=full_actions,
                        full_timesteps=full_timesteps,
                        full_attention_mask=full_attention_mask,
                        style_id=style_id
                    )

    return model


def test(
        model: VariationalStyleDecisionTransformer,
        dataloader: DataLoader,
        epochs=10,
        track=False,
        batch_number=0,
):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss = 0
    n_correct = 0
    n_actions = 0

    pbar = tqdm(range(epochs))
    test_batches_per_epoch = len(dataloader)

    for epoch in pbar:
        for batch, batch_dict in enumerate(dataloader):
            # Unpack batch dictionary
            s = batch_dict["states"]
            a = batch_dict["actions"]
            rtg = batch_dict["returns_to_go"]
            ti = batch_dict["timesteps"]

            full_states = batch_dict["full_states"]
            full_actions = batch_dict["full_actions"]
            full_timesteps = batch_dict["full_timesteps"]
            full_attn_mask = batch_dict["full_attention_mask"]

            if model.transformer_config.time_embedding_type == "linear":
                ti = ti.to(t.float32)
            action_space = model.environment_config.action_space.n
            a[a == -10] = action_space

            output = model(
                states=s,
                actions=a[:, :-1].unsqueeze(-1)
                if a.shape[1] > 1
                else None,
                rtgs=rtg,
                timesteps=ti.unsqueeze(-1),
                full_states=full_states,
                full_actions=full_actions,
                full_timesteps=full_timesteps,
                full_attn_masks=full_attn_mask,
            )
            action_preds = output['action_preds']

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


def get_dataloaders(trajectory_data_set, offline_config, collate_fn):
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
        collate_fn=collate_fn,
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
        collate_fn=collate_fn,
    )

    return train_dataloader, test_dataloader

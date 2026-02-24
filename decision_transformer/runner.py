import json
import os
import time
import warnings
from typing import Callable
import torch as t
import wandb

from configs import RunConfig, TransformerModelConfig, OfflineTrainConfig, \
    ConfigJsonEncoder, EnvironmentConfig
from dataset_utils.hard_prompt_dataset import HardPromptDataset
from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset, TrajectoryVisualizer, \
    one_hot_encode_observation
from dataset_utils.soft_prompt_dataset import SoftPromptDataset
from dataset_utils.soft_prompt_enc_dec_dataset import SoftPromptEncDecDataset
from decision_transformer.configs import TransformerEncoderConfig
from decision_transformer.prompting_decision_transformer import PromptingDecisionTransformer
from decision_transformer.style_enc_decision_transformer_dec import VariationalStyleDecisionTransformer
from decision_transformer.train_enc_dec import train_enc_dec
from train import train
from style_decision_transformer import StyleDecisionTransformer
from util import get_max_len_from_model_type


def run_decision_transformer(
        run_config: RunConfig,
        transformer_config: TransformerModelConfig,
        offline_config: OfflineTrainConfig,
        encoder_config: TransformerEncoderConfig,
        make_env: Callable,
):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    device = set_device(run_config)

    if offline_config.trajectory_paths is None:
        raise ValueError("Must specify a trajectory path.")

    max_len = get_max_len_from_model_type(
        offline_config.model_type, transformer_config.n_ctx
    )

    preprocess_observations = (
        None
        if not offline_config.convert_to_one_hot
        else one_hot_encode_observation
    )
    if offline_config.soft_prompt_mode is None:
        trajectory_data_set = TrajectoryDataset(
            trajectory_paths=offline_config.trajectory_paths,
            max_len=max_len,
            pct_traj=offline_config.pct_traj,
            prob_go_from_end=offline_config.prob_go_from_end,
            device=device,
            preprocess_observations=preprocess_observations,
        )
    else:
        if offline_config.soft_prompt_mode:
            trajectory_data_set = SoftPromptDataset(
                trajectory_paths=offline_config.trajectory_paths,
                vae_model_path=offline_config.vae_model_save_path,
                vae_model_type=offline_config.vae_model_type,
                vae_model_parameters=offline_config.vae_model_params,
                max_len=max_len,
                pct_traj=offline_config.pct_traj,
                prob_go_from_end=offline_config.prob_go_from_end,
                device=device,
                preprocess_observations=preprocess_observations,
            )
        else:
            if offline_config.soft_prompt_enc_dec_mode:
                trajectory_data_set = SoftPromptEncDecDataset(
                    trajectory_paths=offline_config.trajectory_paths,
                    vae_model_path=offline_config.vae_model_save_path,
                    vae_model_type=offline_config.vae_model_type,
                    vae_model_parameters=offline_config.vae_model_params,
                    max_len=max_len,
                    pct_traj=offline_config.pct_traj,
                    prob_go_from_end=offline_config.prob_go_from_end,
                    device=device,
                    preprocess_observations=preprocess_observations,
                )
            else:
                trajectory_data_set = HardPromptDataset(
                    trajectory_paths=offline_config.trajectory_paths,
                    max_len=max_len,
                    max_prompt_len=transformer_config.prompt_len,
                    pct_traj=offline_config.pct_traj,
                    prob_go_from_end=offline_config.prob_go_from_end,
                    device=device,
                    preprocess_observations=preprocess_observations,
                )

    # make an environment
    trajectory_data_set.metadata["args"]["env_id"] = "MiniGrid-three_style"
    env_id = trajectory_data_set.metadata["args"]["env_id"]
    # pretty print the metadata

    if "view_size" not in trajectory_data_set.metadata["args"]:
        trajectory_data_set.metadata["args"]["view_size"] = 7

    environment_config = EnvironmentConfig(
        env_id=trajectory_data_set.metadata["args"]["env_id"],
        one_hot_obs=trajectory_data_set.observation_type == "one_hot",
        view_size=trajectory_data_set.metadata["args"]["view_size"],
        fully_observed=False,
        capture_video=False,
        render_mode="rgb_array",
    )

    wandb_args = (
            run_config.__dict__
            | transformer_config.__dict__
            | offline_config.__dict__
    )

    if run_config.track:
        wandb.login(key="567347e5e788f2ecb2b9f6412ce418df1a4e41c2")
        run_name = f"{env_id}__{run_config.exp_name}__{run_config.seed}__{int(time.time())}"
        wandb.init(
            project=run_config.wandb_project_name,
            entity=run_config.wandb_entity,
            name=run_name,
            config=wandb_args,
            # save_code=True
        )
        trajectory_visualizer = TrajectoryVisualizer(trajectory_data_set)
        # fig = trajectory_visualizer.plot_reward_over_time()
        # wandb.log({"dataset/reward_over_time": wandb.Plotly(fig)})
        # fig = trajectory_visualizer.plot_base_action_frequencies()
        # wandb.log({"dataset/base_action_frequencies": wandb.Plotly(fig)})
        # wandb.log(
        #     {"dataset/num_trajectories": trajectory_data_set.num_trajectories}
        # )
    if offline_config.soft_prompt_mode:
        model = StyleDecisionTransformer(
            environment_config=environment_config,
            transformer_config=transformer_config,
            device=device,
        )
    elif offline_config.soft_prompt_enc_dec_mode:
        encoder_config.pos_max_len = trajectory_data_set.max_seq_len

        model = VariationalStyleDecisionTransformer(
            environment_config=environment_config,
            transformer_config=transformer_config,
            encoder_config=encoder_config,
            device=device,
        )
    else:
        model = PromptingDecisionTransformer(
            environment_config=environment_config,
            transformer_config=transformer_config,
            device=device,
        )


    if offline_config.soft_prompt_enc_dec_mode:
        model = train_enc_dec(
            model=model,
            trajectory_data_set=trajectory_data_set,
            make_env=make_env,
            device=device,
            offline_config=offline_config,
        )
    else:
        model = train(
            model=model,
            trajectory_data_set=trajectory_data_set,
            make_env=make_env,
            device=device,
            offline_config=offline_config,
        )

    if run_config.track:
        # save the model with pickle, then upload it
        # as an artifact, then delete it.
        # name it after the run name.
        if not os.path.exists("models"):
            os.mkdir("models")

        model_path = f"models/{run_name}.pt"

        store_transformer_model(
            path=model_path,
            model=model,
            offline_config=offline_config,
        )

        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        # os.remove(model_path)

        wandb.finish()


def store_transformer_model(path, model, offline_config):
    t.save(
        {
            "model_state_dict": model.state_dict(),
            "offline_config": json.dumps(
                offline_config, cls=ConfigJsonEncoder
            ),
            "environment_config": json.dumps(
                model.environment_config, cls=ConfigJsonEncoder
            ),
            "model_config": json.dumps(
                model.transformer_config, cls=ConfigJsonEncoder
            ),
        },
        path,
    )


def set_device(run_config):
    if run_config.device == str(t.device("cuda")):
        if t.cuda.is_available():
            device = t.device("cuda")
        else:
            print("CUDA not available, using CPU instead.")
            device = t.device("cpu")
    elif run_config.device == t.device("cpu"):
        device = t.device("cpu")
    elif run_config.device == t.device("mps"):
        if t.mps.is_available():
            device = t.device("mps")
        else:
            print("MPS not available, using CPU instead.")
            device = t.device("cpu")
    else:
        print("Invalid device, using CPU instead.")
        device = t.device("cpu")

    return device

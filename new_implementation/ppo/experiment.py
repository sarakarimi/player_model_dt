#Test 
from new_implementation.configs import (
    EnvironmentConfig,
    OnlineTrainConfig,
    RunConfig,
)
from runner import ppo_runner
from util import parse_args

if __name__ == "__main__":
    args = parse_args()

    run_config = RunConfig(
        exp_name=args.exp_name,
        seed=args.seed,
        device="cuda" if args.cuda else "cpu",
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
    )

    environment_config = EnvironmentConfig(
        env_id=args.env_id,
        one_hot_obs=args.one_hot_obs,
        fully_observed=args.fully_observed,
        max_steps=args.max_steps,
        capture_video=args.capture_video,
        view_size=args.view_size,
        device=run_config.device,
        env_mode=args.mode
    )

    online_config = OnlineTrainConfig(
        hidden_size=args.hidden_size,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        decay_lr=args.decay_lr,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        trajectory_path=args.trajectory_path,
        fully_observed=args.fully_observed,
        num_checkpoints=args.num_checkpoints,
        device=run_config.device,
    )

    ppo_runner(
        run_config=run_config,
        environment_config=environment_config,
        online_config=online_config,
    )

import torch
import wandb
import argparse
import random
from torch.nn import functional as F

from decision_transformers.official_implementation.vector_obs.util_functions import eval_and_gif_multi_env
from decision_transformers.official_implementation.vector_obs.models.bimodal_decision_transformer import BimodalDecisionTransformer
from decision_transformers.official_implementation.vector_obs.training.bimodal_seq_trainer import BimodalSequenceTrainer
from decision_transformers.official_implementation.vector_obs.util_functions import make_bimodal_dataset, process_info, get_batch, \
    get_mode_batch, gen_env, get_env_list, eval_episodes, get_mode, load_checkpoint
from envs.double_goal_minigrid import DoubleGoalEnv
from utils.minigrid_wrappers import FullyObsFeatureWrapper


def bimodal_experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', True)

    env_name = variant['env']
    env_mode = variant["env_mode"]
    env = gen_env(env_mode)

    max_ep_len = 27  # 50 # 100
    env_targets = [0.75, 0.87, 1]
    scale = 1.
    state_dim = 300  # env.observation_space.shape[0]
    act_dim = 1


    K = variant['K']
    batch_size = variant['per_env_batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')

    # save trained model and wandb metrics
    dataset_name = "long_uni_modal_restricted"
    group_name = f'{env_name}-{dataset_name}-dataset'
    exp_prefix = f'mode-{env_mode}-vocab_size-{K}-max_episode_len-{max_ep_len}-exp_id-{random.randint(int(1e5), int(1e6) - 1)}'
    model_name = f'dt_double_goal-mode-{env_mode}-vocab_size-{K}-max_episode_len-{max_ep_len}-env-{env_name}-dataset-{dataset_name}'
    checkpoint_path = f"/home/sara/repositories/player_model_dt/trained_models/{model_name}.pth"


    # load training dataset
    mode_list = [1, 2]
    trajectories_list = make_bimodal_dataset(mode_list)

    info, env_list = get_env_list(mode_list, max_ep_len, env_targets, scale, state_dim, act_dim, device)
    info = process_info(mode_list, trajectories_list, info, mode, pct_traj, variant)

    model = BimodalDecisionTransformer(
        state_dim=state_dim,
        act_dim=4,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    trainer = BimodalSequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: F.cross_entropy(a_hat, a),
        eval_fns=None,
        get_prompt_batch=get_mode_batch(trajectories_list, True, info, variant, mode_list)
    )
    if not variant['evaluation']:
        if log_to_wandb:
            wandb.init(
                name=exp_prefix,
                group=group_name,
                project='prompt-decision-transformer',
                config=variant
            )

        for i in range(variant['max_iters']):
            outputs = trainer.pure_train_iteration_mix(
                num_steps=variant['num_steps_per_iter'],
                no_prompt=args.no_prompt
            )
            if i % args.train_eval_interval == 0:
                # evaluate train
                env = FullyObsFeatureWrapper(
                    DoubleGoalEnv(
                        mode=0,
                        render_mode="rgb_array",
                        agent_start_pos=None,
                        max_steps=50,
                    )
                )
                train_eval_logs = trainer.eval_iteration_multienv(
                    get_mode,
                    eval_episodes, mode_list, info, variant, [env], iter_num=i + 1,
                    print_logs=True, no_prompt=args.no_prompt, group='train')
                outputs.update(train_eval_logs)

            if i % variant['save_interval'] == 0:
                trainer.save_model(checkpoint_path=checkpoint_path)

            outputs.update({"global_step": i}) # set global step as iteration

            if log_to_wandb:
                wandb.log(outputs)

    if variant['evaluation']:
        env = FullyObsFeatureWrapper(
            DoubleGoalEnv(
                mode=1,
                render_mode="rgb_array",
                agent_start_pos=None,
                max_steps=50,
            )
        )
        model = load_checkpoint(model, checkpoint_path)
        eval_and_gif_multi_env(model, variant['max_iters'], model_name, max_ep_len, get_mode, [1, 2],
                               info, variant, env, target_return=[0.98], no_prompt=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='miniGrid')
    parser.add_argument('--env_mode', type=int, default=0)
    parser.add_argument('--prompt-episode', type=int, default=1)
    parser.add_argument('--prompt-length', type=int, default=5)
    parser.add_argument('--stochastic-prompt', action='store_true', default=True)
    parser.add_argument('--no-prompt', action='store_true', default=False)
    parser.add_argument('--no-r', action='store_true', default=False)
    parser.add_argument('--no-rtg', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--finetune_steps', type=int, default=10)
    parser.add_argument('--finetune_batch_size', type=int, default=256)
    parser.add_argument('--finetune_opt', action='store_true', default=True)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--no_state_normalize', action='store_true', default=False)
    parser.add_argument('--average_state_mean', action='store_true', default=True)
    parser.add_argument('--evaluation', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--load-path', type=str, default=None)  # choose a model when in evaluation mode

    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--per_env_batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)  # 10000*(number of environments)
    parser.add_argument('--num_eval_episodes', type=int, default=50)
    parser.add_argument('--max_iters', type=int, default=50)
    parser.add_argument('--num_steps_per_iter', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--train_eval_interval', type=int, default=5)
    parser.add_argument('--test_eval_interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)

    args = parser.parse_args()
    bimodal_experiment('minigrid-experiment', variant=vars(args))
from ast import parse
import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import time
import itertools
from torch.nn import functional as F

from envs.double_goal_minigrid import DoubleGoalEnv
from envs.minigrid_wrappers import FullyObsFeatureWrapper
from official_implementation.vector_obs.models.prompt_decision_transformer import PromptDecisionTransformer
from official_implementation.vector_obs.training.prompt_seq_trainer import PromptSequenceTrainer
from official_implementation.vector_obs.util_functions import get_env_list
from official_implementation.vector_obs.prompt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune
from official_implementation.vector_obs.prompt_utils import process_total_data_mean, load_data_prompt, process_info
from official_implementation.vector_obs.prompt_utils import eval_episodes

from collections import namedtuple
import json, pickle, os

def experiment_mix_env(
        exp_prefix,
        variant,
):
    device = variant['device']
    log_to_wandb = variant['log_to_wandb']

    ######
    # construct train and test environments
    ######
    max_ep_len = 27  # 50 # 100
    env_targets = [0.75, 0.87, 1]
    scale = 1.
    state_dim = 300  # env.observation_space.shape[0]
    act_dim = 1

    cur_dir = os.getcwd()
    config_save_path = os.path.join(cur_dir, 'config')
    data_save_path = ""
    prompt_save_path = ""

    save_path = os.path.join(cur_dir, 'model_saved/')
    if not os.path.exists(save_path): os.mkdir(save_path)

    train_env_name_list, test_env_name_list = [], []
    train_env_name_list = [1, 2]
    test_env_name_list = [1, 2] #.append(args.env)
    # training envs
    info, env_list = get_env_list(train_env_name_list, max_ep_len, env_targets, scale, state_dim, act_dim, device)
    # testing envs
    test_info, test_env_list = get_env_list(test_env_name_list, max_ep_len, env_targets, scale, state_dim, act_dim, device)
    print(f'Env Info: {info} \n\n Test Env Info: {test_info}\n\n\n')
    # print(f'Env List: {env_list} \n\n Test Env List: {test_env_list}')
    ######
    # process train and test datasets
    ######

    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')
    dataset_mode = variant['dataset_mode']
    test_dataset_mode = variant['test_dataset_mode']
    train_prompt_mode = variant['train_prompt_mode']
    test_prompt_mode = variant['test_prompt_mode']

    # load training dataset
    trajectories_list, prompt_trajectories_list = load_data_prompt(train_env_name_list, data_save_path, prompt_save_path)
    # load testing dataset
    test_trajectories_list, test_prompt_trajectories_list = load_data_prompt(test_env_name_list, data_save_path, prompt_save_path)

    # change to total train trajecotry 
    if variant['average_state_mean']:
        train_total = list(itertools.chain.from_iterable(trajectories_list))
        test_total = list(itertools.chain.from_iterable(test_trajectories_list))
        total_traj_list = train_total + test_total
        print(len(total_traj_list))
        total_state_mean, total_state_std= process_total_data_mean(total_traj_list, mode)
        variant['total_state_mean'] = total_state_mean
        variant['total_state_std'] = total_state_std

    # process train info
    info = process_info(train_env_name_list, trajectories_list, info, mode, dataset_mode, pct_traj, variant)
    # process test info
    test_info = process_info(test_env_name_list, test_trajectories_list, test_info, mode, test_dataset_mode, pct_traj, variant)

    ######
    # construct dt model and trainer
    ######
    
    exp_prefix = exp_prefix + '-' + args.env
    num_env = len(train_env_name_list)
    group_name = f'{exp_prefix}-{str(num_env)}-Env-{dataset_mode}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    # state_dim = test_env_list[0].observation_space.shape[0]
    # act_dim = test_env_list[0].action_space.shape[0]

    model = PromptDecisionTransformer(
        state_dim=state_dim,
        act_dim=4, #act_dim,
        max_length=K,
        max_ep_len=1000,
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

    env_name = train_env_name_list[0]
    prompt_trajectories_list = trajectories_list
    trainer = PromptSequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch(trajectories_list[0], info[str(env_name)], variant),
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: F.cross_entropy(a_hat, a),
        eval_fns=None,
        get_prompt=get_prompt(prompt_trajectories_list[0], info[str(env_name)], variant),
        get_prompt_batch=get_prompt_batch(trajectories_list, None, info, variant, train_env_name_list)
    )


    if not variant['evaluation']:
        ######
        # start training
        ######
        if log_to_wandb:
            wandb.init(
                name=exp_prefix,
                group=group_name,
                project='prompt-decision-transformer',
                config=variant
            )
            save_path += wandb.run.name
            os.mkdir(save_path)

        # construct model post fix
        model_post_fix = '_TRAIN_'+variant['train_prompt_mode']+'_TEST_'+variant['test_prompt_mode']
        if variant['no_prompt']:
            model_post_fix += '_NO_PROMPT'
        if variant['finetune']:
            model_post_fix += '_FINETUNE'
        if variant['no_r']:
            model_post_fix += '_NO_R'
        
        for iter in range(variant['max_iters']):
            env_id = iter % num_env
            env_name = train_env_name_list[env_id]
            outputs = trainer.pure_train_iteration_mix(
                num_steps=variant['num_steps_per_iter'], 
                no_prompt=args.no_prompt
                )

            # start evaluation
            # if iter % args.test_eval_interval == 0:
            #     # evaluate test
            #     if not args.finetune:
            #         test_eval_logs = trainer.eval_iteration_multienv(
            #             get_prompt, test_prompt_trajectories_list,
            #             eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=iter + 1,
            #             print_logs=True, no_prompt=args.no_prompt, group='test')
            #         outputs.update(test_eval_logs)
            #     else:
            #         test_eval_logs = trainer.finetune_eval_iteration_multienv(
            #             get_prompt, get_batch_finetune, test_prompt_trajectories_list, test_trajectories_list,
            #             eval_episodes, test_env_name_list, test_info,
            #             variant, test_env_list, iter_num=iter + 1,
            #             print_logs=True, no_prompt=args.no_prompt,
            #             group='finetune-test', finetune_opt=variant['finetune_opt'])
            #         outputs.update(test_eval_logs)

            if iter % args.train_eval_interval == 0:
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
                    get_prompt, prompt_trajectories_list,
                    eval_episodes, train_env_name_list, info, variant, [env], iter_num=iter + 1,
                    print_logs=True, no_prompt=args.no_prompt, group='train')
                outputs.update(train_eval_logs)

            if iter % variant['save_interval'] == 0:
                trainer.save_model(
                    env_name=args.env, 
                    postfix=model_post_fix+'_iter_'+str(iter), 
                    folder=save_path)

            outputs.update({"global_step": iter}) # set global step as iteration

            if log_to_wandb:
                wandb.log(outputs)
        
        trainer.save_model(env_name=args.env,  postfix=model_post_fix+'_iter_'+str(iter),  folder=save_path)

    else:
        ####
        # start evaluating
        ####
        saved_model_path = os.path.join(save_path, variant['load_path'])
        model.load_state_dict(torch.load(saved_model_path))
        print('model initialized from: ', saved_model_path)
        eval_iter_num = int(saved_model_path.split('_')[-1])

        eval_logs = trainer.eval_iteration_multienv(
                    get_prompt, test_prompt_trajectories_list,
                    eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=eval_iter_num, 
                    print_logs=True, no_prompt=args.no_prompt, group='eval')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='miniGrid')
    parser.add_argument('--dataset_mode', type=str, default='expert')
    parser.add_argument('--test_dataset_mode', type=str, default='expert')
    parser.add_argument('--train_prompt_mode', type=str, default='expert')
    parser.add_argument('--test_prompt_mode', type=str, default='expert')

    parser.add_argument('--prompt-episode', type=int, default=1)
    parser.add_argument('--prompt-length', type=int, default=5)
    parser.add_argument('--stochastic-prompt', action='store_true', default=False)
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
    parser.add_argument('--load-path', type=str, default= None) # choose a model when in evaluation mode

    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000*(number of environments)
    parser.add_argument('--num_eval_episodes', type=int, default=50) 
    parser.add_argument('--max_iters', type=int, default=5000) 
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--train_eval_interval', type=int, default=50)
    parser.add_argument('--test_eval_interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)

    args = parser.parse_args()
    experiment_mix_env('minigrid-prompt-experiment', variant=vars(args))
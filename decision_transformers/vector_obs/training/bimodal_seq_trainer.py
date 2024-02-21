# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time
from wandb import env
import copy

from decision_transformers.vector_obs.util_functions import flatten_mode


class BimodalSequenceTrainer:

    def __init__(self, model, optimizer, batch_size, checkpoint_path, loss_fn,
                 scheduler=None, eval_fns=None, get_prompt=None, get_prompt_batch=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        # self.get_mode = get_prompt
        self.prompt = None #self.get_mode()  # sample prompt data when initialization
        self.get_prompt_batch = get_prompt_batch

        self.start_time = time.time()

    def pure_train_iteration_mix(self, num_steps, no_prompt=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step_mix(no_prompt)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs

    def train_step_mix(self, no_prompt=False):
        mode, batch = self.get_prompt_batch()
        states, actions, rewards, dones, rtg, timesteps, attention_mask = batch
        action_target = torch.clone(actions)
        if no_prompt:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask, mode=None
            )
        else:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask, mode=mode
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1)[attention_mask.reshape(-1) > 0]

        action_target = action_target.long()
        action_preds = action_preds

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        # with torch.no_grad():
        #     self.diagnostics['training/action_error'] = torch.mean(
        #         (action_preds - action_target) ** 2).detach().cpu().item()

        return loss.detach().cpu().item()

    def eval_iteration_multienv(self, get_mode, eval_episodes, env_name_list, info,
                                variant, env_list, iter_num=0, print_logs=False, no_prompt=False, group='test'):
        print('evaluate at tasks: ', env_name_list)
        logs = dict()
        print('start evaluating...')
        self.model.eval()

        eval_start = time.time()
        for env_id, env_name in enumerate(env_name_list):

            # need to sample eval_fn and prompt together
            self.eval_fns = [eval_episodes(tar, info[str(env_name)], variant, env_list[0], env_name) for tar in
                             info[str(env_name)]['env_targets']]
            get_mode_fn = get_mode(info[str(env_name)], env_id, variant)
            if not no_prompt:
                self.prompt = flatten_mode(get_mode_fn(), batch_size=1)
            else:
                self.prompt = None
            for eval_fn in self.eval_fns:
                # print('env_name : ', env_list[env_id])
                outputs = eval_fn(self.model, prompt=self.prompt)
                for k, v in outputs.items():
                    logs[f'{group}-evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def save_model(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)  # model save
        print('model saved to ', checkpoint_path)

# if __name__ == '__main__':
#
#     prompt_dataset_path = "/home/sara/repositories/player_model_dt/1/cheetah_dir-0-prompt-expert.pkl"
#     with open(prompt_dataset_path, 'rb') as f:
#         prompt_trajectories = pickle.load(f)
#         print(prompt_trajectories)
#     max_len = 5
#
#     def fn(sample_size=1):
#         # random sample prompts with fixed length (prompt-length) in num episodes (prompt-episode)
#         batch_inds = np.random.choice(
#             np.arange(len(prompt_trajectories)),
#             size=int(1 * sample_size),
#             replace=True,
#             # p=p_sample,  # reweights so we sample according to timesteps
#         )
#
#         s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
#         for i in range(int(1 * sample_size)):
#             # if variant["stochastic_prompt"]:
#             traj = prompt_trajectories[int(batch_inds[i])]  # random select traj
#             print(len(traj), batch_inds, i)
#             # else:
#                 # traj = prompt_trajectories[int(sorted_inds[-i])]  # select the best traj with highest rewards
#                 # traj = prompt_trajectories[i]
#             si = max(0, traj['rewards'].shape[0] - max_len - 1)  # select the last traj with length max_len
#             print(si, max_len)
#             # get sequences from dataset
#             print(len(traj['rewards']))
#             r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
#             print(r)
#             # padding and state + reward normalization
#
#         return r
#
#     fn(16)
#
#
#

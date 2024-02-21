# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md
import imageio
import numpy as np
import torch


def prompt_evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        prompt=None,
        no_r=False,
        no_rtg=False,
        no_state_normalize=False
):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()[0]
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # print('evaluate/t', t)
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        if no_state_normalize:
            action = model.get_action(
                states.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                prompt=prompt
            )
        else:
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                mode=prompt
            )

        actions[-1] = action
        # action = action.detach().cpu().numpy()
        #
        # state, reward, done, infos = env.step(action)
        state, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0, -1])

        state = state.flatten()
        done = terminated or truncated
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        if no_r:
            rewards[-1] = 0.0

        if mode != 'delayed':
            pred_return = target_return[0, -1] - (reward / scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        if no_rtg:
            target_return = torch.ones_like(target_return) * ep_return
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return


def eval_and_gif(env, model, eval_iter, model_name, target_ret, state_mean, state_std, max_ep_len, task, prompt):
    device = "cuda"
    max_ep_len = max_ep_len
    state_dim = 300  # env.observation_space.shape[0]
    act_dim = 1
    images = []

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    for i in range(eval_iter):
        state = env.reset()[0]
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        ep_return = target_ret
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        episode_return, episode_length = 0, 0

        # Get GIF of agent performance
        img = env.render()
        blank_screen = np.zeros(shape=img.shape, dtype=np.uint8)
        images.append(np.asarray(blank_screen))
        for t in range(max_ep_len):
            img = env.render()
            images.append(img)

            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                mode=prompt
            )
            actions[-1] = action

            state, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0, -1])
            # print("reward", reward)
            state = state.flatten()
            done = terminated or truncated
            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0, -1]  # - (reward/scale)

            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                images.append(np.asarray(blank_screen))
                break

    imageio.mimsave(
        f"/home/sara/repositories/player_model_dt/gifs/" + model_name + str(task) + ".gif",
        [np.array(img) for i, img in enumerate(images)],
        fps=4,
    )



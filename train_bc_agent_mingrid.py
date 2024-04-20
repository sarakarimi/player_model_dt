import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import imageio
from envs.double_goal_minigrid import DoubleGoalEnv
from utils.minigrid_wrappers import FullyObsFeatureWrapper


class BehaviorCloningModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BehaviorCloningModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


class BehaviorCloningDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observation = self.observations[idx]
        action = self.actions[idx]
        return observation, action


def count_actions(actions):
    unique, counts = np.unique(actions, return_counts=True)
    action_counts = dict(zip(unique, counts))
    return action_counts


def load_dataset(file_path):
    data = np.load(file_path)
    observations = data['obs']
    actions = data['action']

    # Assuming 'obs' are stored in CHW format ([3, 10, 10]) and we need them in HWC format ([10, 10, 3])
    # Transpose each observation
    observations = observations.transpose((0, 2, 3, 1))
    return observations, actions


def evaluate_agent(env, model, num_episodes=1000):
    total_rewards = 0
    model.eval()
    for _ in range(num_episodes):
        obs = env.reset()
        obs = obs[0].reshape(-1)
        done = False
        episode_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action_probs = model.predict(obs_tensor)
            action = torch.argmax(action_probs, dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = obs.reshape(-1)
            done = terminated or truncated
            episode_reward += reward

        total_rewards += episode_reward

    return total_rewards / num_episodes


def save_agent_trajectories_gif(env, model, gif_name="agent_trajectories.gif"):
    model.eval()
    # Get GIF of agent performance
    images = []
    obs = env.reset()
    obs = obs[0].reshape(-1)
    img = env.render()
    blank_screen = np.zeros(shape=img.shape, dtype=np.uint8)
    images.append(np.asarray(blank_screen))
    for i in range(100):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        images.append(img)
        action_probs = model.predict(obs_tensor)
        action = torch.argmax(action_probs, dim=1).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = obs.reshape(-1)
        done = terminated or truncated
        if done:
            images.append(np.asarray(blank_screen))
            obs = env.reset()
            obs = obs[0].reshape(-1)
        img = env.render()
    imageio.mimsave(
        gif_name,
        [np.array(img) for i, img in enumerate(images)],
        fps=4,
    )

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
observations, actions = load_dataset("data/double_goal_mode_1_image_ft_0_nsteps_100000_ppo_num_episodes10000_eps_0.0_gamma_0.99_dr_False.npz")

# Print action counts
action_counts = count_actions(actions)
print("Action Counts:", action_counts)

# Convert numpy arrays to PyTorch tensors and move to device
obs_torch = torch.tensor(observations, dtype=torch.float32).to(device)
action_torch = torch.tensor(actions, dtype=torch.float32).to(device)

# Calculate the number of actions
num_actions = 4
obs_size = 300

# Create a dataset and dataloader
dataset = BehaviorCloningDataset(obs_torch, action_torch)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the behavior cloning model and move to device
model = BehaviorCloningModel(input_size=obs_size, output_size=num_actions).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the MiniGrid environment for evaluation
env = FullyObsFeatureWrapper(
            DoubleGoalEnv(
                mode=1,
                render_mode="rgb_array",
                agent_start_pos=None,
                max_steps=50,
            )
        )

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_obs, batch_action in dataloader:
        # Move batch to device
        batch_obs, batch_action = batch_obs.to(device), batch_action.to(device)
        action_indices = batch_action.long().squeeze()

        # Forward pass
        outputs = model(batch_obs)

        # Compute loss
        loss = criterion(outputs, action_indices)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_epoch_loss = epoch_loss / len(dataloader)

    # Evaluate the agent on the environment
    avg_reward = evaluate_agent(env, model)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss}, Avg Reward: {avg_reward}")

    # Save agent trajectories GIF every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_agent_trajectories_gif(env, model, gif_name=f"trained_models/bc_agent_trajectories_epoch_{epoch + 1}.gif")

# Save the trained model
torch.save(model.state_dict(), "trained_models/behavior_cloning_model.pth")

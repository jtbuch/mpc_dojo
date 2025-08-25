import gym
import numpy as np
import mujoco
import cv2
import copy
from collections import defaultdict

# ----- MuJoCo Environment Wrapper -----
class MuJoCoWorldWrapper:
    def __init__(self, env_name, n_action_bins=5):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.obs, _ = self.env.reset()
        self.done = False

        # Discretize action space
        self.action_grid = self.discretize_action_space(self.env.action_space, n_action_bins)
        self.n_actions = len(self.action_grid)

    def discretize_action_space(self, action_space, n_bins):
        low, high = action_space.low, action_space.high
        grids = [np.linspace(l, h, n_bins) for l, h in zip(low, high)]
        return np.array(np.meshgrid(*grids)).T.reshape(-1, len(low))

    def reset(self):
        self.obs, _ = self.env.reset()
        return self.obs

    def transition(self, obs, action):
        qpos = np.copy(self.env.unwrapped.data.qpos)
        qvel = np.copy(self.env.unwrapped.data.qvel)
        self.env.unwrapped.data.qpos[:] = qpos
        self.env.unwrapped.data.qvel[:] = qvel
        mujoco.mj_forward(self.env.unwrapped.model, self.env.unwrapped.data)
        obs_, reward, done, truncated, _ = self.env.step(action)
        return obs_, reward

    def render_frame(self):
        return self.env.render()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

# ----- RL Bird for MuJoCo -----
class RLShittyBirdMuJoCo:
    def __init__(self, obs_dim, action_grid, gamma=0.9, alpha=0.05, epsilon=0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.action_grid = action_grid
        self.n_actions = len(action_grid)
        self.q_table = defaultdict(lambda: np.random.randn(self.n_actions))
        self.encoder = lambda obs: tuple(np.round(obs, decimals=2))

    def policy(self, obs):
        state = self.encoder(obs)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, obs, action_idx, reward, next_obs, next_action_idx):
        s = self.encoder(obs)
        s_ = self.encoder(next_obs)
        Q = self.q_table
        Q_s_a = Q[s][action_idx]
        Q_s1_a1 = Q[s_][next_action_idx]
        Q[s][action_idx] = (1 - self.alpha) * Q_s_a + self.alpha * (reward + self.gamma * Q_s1_a1)

    def register_action(self):
        pass

# ----- Simulation Loop -----
def simulate(bird, world, n_episodes, time_steps, train=False, save_video=False, video_path="sim_output.mp4"):
    if save_video:
        frame = world.render_frame()
        h, w = frame.shape[:2]
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MP4V"), 30, (w, h))

    for ep in range(n_episodes):
        obs = world.reset()
        a_idx = bird.policy(obs)
        bird.register_action()

        for t in range(time_steps):
            action = world.action_grid[a_idx]
            next_obs, reward = world.transition(obs, action)
            next_a_idx = bird.policy(next_obs)
            bird.register_action()

            if train:
                bird.update(obs, a_idx, reward, next_obs, next_a_idx)

            if save_video:
                frame = world.render_frame()
                video_writer.write(frame)

            obs, a_idx = next_obs, next_a_idx

    if save_video:
        video_writer.release()
    world.close()

# ----- Main Execution -----
if __name__ == "__main__":
    # Setup world and bird
    world = MuJoCoWorldWrapper("InvertedPendulum-v5", n_action_bins=3)
    bird = RLShittyBirdMuJoCo(obs_dim=len(world.obs), action_grid=world.action_grid)

    # Simulate
    simulate(bird, world, n_episodes=1, time_steps=200, train=True, save_video=True, video_path="pendulum_bird.mp4")

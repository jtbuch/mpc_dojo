from configs import *

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
#from casadi import vertcat
from scipy.linalg import solve_discrete_are
import imageio

plt.ion()
plt.close('all')


def simulate(bird, n_episodes, time_steps):
    """ Simulate bird, possible episodic SARSA. """

    # Initialize trajectory tracking [episode, time_step, SAR]
    hist = np.zeros((n_episodes, time_steps, 3), dtype=int)

    # Episodic loop
    for epi in range(n_episodes):


        # Create the environment
        env = gym.make("InvertedPendulum-v5", render_mode="rgb_array", reset_noise_scale=0.0001) 
        env.reset()
        writer = imageio.get_writer("output.gif", fps=30)

        angles   = np.zeros((time_steps))
        actions  = np.zeros((time_steps))
        rewards  = np.zeros((time_steps))

        for t in range(time_steps):
            print(f"States: {env.unwrapped.data.qpos}, {env.unwrapped.data.qvel}")
            action = bird.mujoco_policy(env)
            actions[t] = action

            obs, reward, done, truncated, info = env.step(np.array([action]))

            angles[t] = obs[1]
            rewards[t] = reward

            if epi == n_episodes - 1:
                frame = env.render()  # Returns a NumPy RGB array (H, W, 3)
                frame = np.asarray(frame).astype(np.uint8)
                writer.append_data(frame)

            # Learning
            # if train: bird.update(state, action, reward, next_state, next_action)

        # Release the VideoWriter object and close the environment
        writer.close()
        env.close()

    return angles, actions


bird = MPCShittyBird(n_actions = 10, planning_width=100, n_planning=10)

angle, actions = simulate(bird, n_episodes=1, time_steps=100)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(angle)
plt.title('Angle')

plt.subplot(2, 1, 2)
plt.plot(actions)
plt.title('Actions')
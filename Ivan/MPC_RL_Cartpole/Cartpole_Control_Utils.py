import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder
import os 
import do_mpc
from casadi import vertcat
from scipy.linalg import solve_discrete_are

class MPCController:
    def __init__(self, horizon=10, dt=0.02, linear=False, recompute_every=1):
        self.horizon = horizon
        self.dt = dt
        self.linear = linear  # New parameter
        self.recompute_every = recompute_every
        self.force_mag = 10.0
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length

        # Define model
        model_type = "continuous"
        self.model = do_mpc.model.Model(model_type)
        
        x = self.model.set_variable(var_type='_x', var_name='x', shape=(1,1))
        x_dot = self.model.set_variable(var_type='_x', var_name='x_dot', shape=(1,1))
        theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
        theta_dot = self.model.set_variable(var_type='_x', var_name='theta_dot', shape=(1,1))
        u = self.model.set_variable(var_type='_u', var_name='u', shape=(1,1))
        
        if self.linear:
            # Linearized dynamics (θ ≈ 0)
            temp = u / self.total_mass  # Ignore θ_dot² term
            thetaacc = (self.gravity * theta - temp) / (
                self.length * (4.0/3.0 - self.masspole/self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc / self.total_mass
        else:
            # Original nonlinear dynamics
            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            temp = (u + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
            )
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        self.model.set_rhs('x', x_dot)
        self.model.set_rhs('x_dot', xacc)
        self.model.set_rhs('theta', theta_dot)
        self.model.set_rhs('theta_dot', thetaacc)
        
        self.model.setup()
        
        # Configure MPC based on linearity
        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_horizon': self.horizon,
            't_step': self.dt,
            'n_robust': 0,
            'state_discretization': 'discrete' if self.linear else 'collocation',
            'collocation_type': 'radau' if not self.linear else None,
            'collocation_deg': 2 if not self.linear else None,
            'collocation_ni': 1 if not self.linear else None,
            'store_full_solution': False,
        }
        self.mpc.set_param(**{k: v for k, v in setup_mpc.items() if v is not None})
       
        self.mpc.set_objective(mterm=theta**2 + x**2, lterm=theta**2 + x**2 + 0.01*u**2)
        self.mpc.bounds['lower','_u','u'] = -self.force_mag
        self.mpc.bounds['upper','_u','u'] = self.force_mag
        
        self.mpc.setup()

    def get_action(self, obs):
        self.mpc.x0 = np.array(obs).reshape(-1, 1)
        self.mpc.set_initial_guess()
        u_opt = self.mpc.make_step(self.mpc.x0)
        return float(u_opt[0])

class LQRController:
    def __init__(self, horizon=10, dt=0.02, linear=False, recompute_every=1):
        # Parameters (some may be ignored for LQR)
        self.dt = dt
        self.force_mag = 10.0
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length

        # Linearized dynamics matrices (continuous-time)
        denominator_theta = self.length * (4/3 - self.masspole / self.total_mass)
        a23 = - (self.polemass_length * self.gravity) / (self.total_mass * denominator_theta)
        a43 = self.gravity / denominator_theta
        b21 = (1/self.total_mass) + (self.polemass_length) / (self.total_mass**2 * denominator_theta)
        b41 = -1 / (self.total_mass * denominator_theta)

        A = np.array([
            [0, 1, 0, 0],
            [0, 0, a23, 0],
            [0, 0, 0, 1],
            [0, 0, a43, 0]
        ])
        B = np.array([[0], [b21], [0], [b41]])

        # Discretize using Euler approximation
        A_d = np.eye(4) + A * self.dt
        B_d = B * self.dt

        # Cost matrices (match MPC's objective)
        Q = np.diag([1.0, 0.0, 1.0, 0.0])  # Penalize x and theta
        R = np.array([[0.01]])              # Penalize control effort

        # Solve Discrete Algebraic Riccati Equation
        P = solve_discrete_are(A_d, B_d, Q, R)
        self.K = np.linalg.inv(B_d.T @ P @ B_d + R) @ (B_d.T @ P @ A_d)

    def get_action(self, obs):
        x = np.array(obs).reshape(-1, 1)  # Convert to column vector
        u = -self.K @ x                   # Optimal control law: u = -Kx
        return float(u[0, 0])


def evaluate_rl_models(rl_train_steps, results_folder="Results/PerformanceResults/", 
                       num_episodes=20, noise_scale=None, seed=42):
    """Evaluate PPO and DQN models under noise."""
    if noise_scale is None:
        noise_scale = np.array([0.5, 0.5, 0.05, 0.05])
    os.makedirs(results_folder, exist_ok=True)
    video_dir = os.path.join("Results", "Videos")
    os.makedirs(video_dir, exist_ok=True)

    # Load models
    ppo_model = PPO.load(f"{results_folder}/ppo_cartpole_model_training_steps_{rl_train_steps}")
    dqn_model = DQN.load(f"{results_folder}/dqn_cartpole_model_training_steps_{rl_train_steps}")

    # Evaluate
    ppo_lengths = _evaluate_rl_model(ppo_model, "ppo", num_episodes, noise_scale, seed, video_dir, rl_train_steps)
    dqn_lengths = _evaluate_rl_model(dqn_model, "dqn", num_episodes, noise_scale, seed, video_dir, rl_train_steps)

    # Save results
    noise_str = str(noise_scale).replace(' ', '_').replace('.', '_')
    np.savetxt(f"{results_folder}/ppo_episode_lengths_steps_{rl_train_steps}_noise_{noise_str}.csv", ppo_lengths, delimiter=",")
    np.savetxt(f"{results_folder}/dqn_episode_lengths_steps_{rl_train_steps}_noise_{noise_str}.csv", dqn_lengths, delimiter=",")
    return ppo_lengths, dqn_lengths

def _evaluate_rl_model(model, model_name, num_episodes, noise_scale, seed, video_dir, rl_train_steps):
    """Helper to evaluate a single RL model."""
    episode_lengths = []
    for episode in range(num_episodes):
        if episode == 0:
            env = gym.make("CartPole-v1", render_mode="rgb_array")
            model_video_dir = os.path.join(video_dir, model_name)
            os.makedirs(model_video_dir, exist_ok=True)
            video_path = f"{model_video_dir}/rl_steps_{rl_train_steps}_ep1_noise_{str(noise_scale).replace('.', '_')}.mp4"
            env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda _: True)
        else:
            env = gym.make("CartPole-v1", render_mode=None)
        obs, _ = env.reset(seed=seed + episode)
        length = 0
        done = False
        while not done and length < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            obs += np.random.default_rng(seed + episode + length).normal(0, noise_scale)
            length += 1
        episode_lengths.append(length)
        env.close()
    return episode_lengths

def evaluate_mpc_controllers(horizons, recompute_intervals, results_folder="Results/PerformanceResults/", 
                             num_episodes=20, noise_scale=None, seed=42, linear=True):
    """Evaluate MPC with various horizons and recompute intervals."""
    if noise_scale is None:
        noise_scale = np.array([0.5, 0.5, 0.05, 0.05])
    os.makedirs(results_folder, exist_ok=True)
    video_dir = os.path.join("Results", "Videos", "mpc")
    os.makedirs(video_dir, exist_ok=True)
    results = {}
    for h in horizons:
        for e in recompute_intervals:
            key = f"h_{h}_e_{e}"
            results[key] = []
            mpc = MPCController(horizon=h, recompute_every=e, linear=linear)
            for ep in range(num_episodes):
                if ep == 0:
                    env = gym.make("CartPole-v1", render_mode="rgb_array")
                    ep_video_dir = os.path.join(video_dir, f"h_{h}_e_{e}_noise_{str(noise_scale).replace('.', '_')}")
                    os.makedirs(ep_video_dir, exist_ok=True)
                    env = gym.wrappers.RecordVideo(env, f"{ep_video_dir}/episode_1.mp4", episode_trigger=lambda _: True)
                else:
                    env = gym.make("CartPole-v1", render_mode=None)
                obs, _ = env.reset(seed=seed + ep)
                length, step = 0, 0
                done = False
                while not done and length < 500:
                    if step % e == 0:
                        action = 1 if mpc.get_action(obs) > 0 else 0
                    obs, _, done, _, _ = env.step(action)
                    obs += np.random.default_rng(seed + ep + length).normal(0, noise_scale)
                    step += 1
                    length += 1
                results[key].append(length)
                env.close()
            noise_str = str(noise_scale).replace(' ', '_').replace('.', '_')
            np.savetxt(f"{results_folder}/mpc_episode_lengths_{key}_noise_{noise_str}.csv", results[key], delimiter=",")
    return results

def evaluate_lqr_controller(results_folder="Results/PerformanceResults/", num_episodes=20, 
                            noise_scale=None, seed=42, horizon=10, recompute_every=1):
    """Evaluate LQR controller."""
    if noise_scale is None:
        noise_scale = np.array([0.5, 0.5, 0.05, 0.05])
    os.makedirs(results_folder, exist_ok=True)
    video_dir = os.path.join("Results", "Videos", "lqr")
    os.makedirs(video_dir, exist_ok=True)
    key = f"h_{horizon}_e_{recompute_every}"
    lqr = LQRController(horizon=horizon, recompute_every=recompute_every)
    lengths = []
    for ep in range(num_episodes):
        if ep == 0:
            env = gym.make("CartPole-v1", render_mode="rgb_array")
            ep_video_dir = os.path.join(video_dir, f"noise_{str(noise_scale).replace('.', '_')}")
            os.makedirs(ep_video_dir, exist_ok=True)
            env = gym.wrappers.RecordVideo(env, f"{ep_video_dir}/episode_1.mp4", episode_trigger=lambda _: True)
        else:
            env = gym.make("CartPole-v1", render_mode=None)
        obs, _ = env.reset(seed=seed + ep)
        length, step = 0, 0
        done = False
        while not done and length < 500:
            if step % recompute_every == 0:
                action = 1 if lqr.get_action(obs) > 0 else 0
            obs, _, done, _, _ = env.step(action)
            obs += np.random.default_rng(seed + ep + length).normal(0, noise_scale)
            step += 1
            length += 1
        lengths.append(length)
        env.close()
    noise_str = str(noise_scale).replace(' ', '_').replace('.', '_')
    np.savetxt(f"{results_folder}/lqr_episode_lengths_{key}_noise_{noise_str}.csv", lengths, delimiter=",")
    return {key: lengths}

def analyze_performance_results(results_folder="Results/PerformanceResults/", 
                                noise_scale=np.array([0.5, 0.5, 0.05, 0.05]), 
                                num_episodes=20):
    """
    Reads CSV files from the given results folder, filters by noise scale, 
    calculates mean and standard deviation of episode lengths, and plots the results.
    """
    
    # Get all CSV files
    files = [file for file in os.listdir(results_folder) if file.endswith(".csv")]
    
    # Filter files based on noise scale
    noise_str = str(noise_scale).replace('', '_')
    files = [file for file in files if file.endswith(f"_noise_{noise_str}.csv")]
    
    results = []
    for file in files:
        data = np.loadtxt(os.path.join(results_folder, file), delimiter=",")
        mean = np.mean(data)
        std = np.std(data)
        model_name = file[:3]
        model_spec = file.split("_")[3:]
        model_spec = model_spec[:model_spec.index('noise')]
        results.append([model_name, model_spec, mean, std])
    
    # Adjust model specifications
    for i, result in enumerate(results):
        if result[0] in ["dqn", "ppo"]:
            results[i][1] = result[1][3]
        else:
            results[i][1] = "_".join(result[1][:])
    
    # Sort results
    results.sort(key=lambda x: (x[0] != "mpc", x[0]))
    
    # Create DataFrame
    df = pd.DataFrame(results, columns=["Model", "Specification", "Mean Episode Length", "Standard Deviation"])
    df[['Horizon', 'Recompute']] = df['Specification'].str.extract(r'h_(\d+)_e_(\d+)').astype('Int64')
    df = df.sort_values(['Horizon', 'Recompute']).reset_index(drop=True)
    
    # Split data
    mpc_df = df[df['Model'] == 'mpc'].copy()
    other_df = df[df['Model'].isin(['dqn', 'ppo'])].copy()
    lqr_df = df[df['Model'] == 'lqr'].copy()
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6), sharey=True, gridspec_kw={'width_ratios': [3, 2, 1]})
    fig.suptitle(f"Performance with noise = {tuple(noise_scale)}; N = {num_episodes} episodes", fontsize=14, fontweight='bold')
    
    # MPC Plot
    cmap_mpc = plt.cm.viridis
    recompute_vals = sorted(mpc_df['Recompute'].unique())
    norm_mpc = plt.Normalize(vmin=min(recompute_vals), vmax=max(recompute_vals))
    for recompute in recompute_vals:
        subset = mpc_df[mpc_df['Recompute'] == recompute].sort_values('Horizon')
        ax1.errorbar(subset['Horizon'], subset['Mean Episode Length'], yerr=subset['Standard Deviation'],
                     label=str(recompute), color=cmap_mpc(norm_mpc(recompute)), marker='o', capsize=4, linestyle='--')
    ax1.set_title('MPC Performance')
    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('Mean Episode Length (± SD)')
    ax1.legend(title='Recompute')
    
    # RL Models Plot
    other_df['Specification'] = pd.to_numeric(other_df['Specification'])
    specs = sorted(other_df['Specification'].unique())
    model_styles = {'dqn': {'color': 'C0', 'label': 'DQN'}, 'ppo': {'color': 'C1', 'label': 'PPO'}}
    for model in ['dqn', 'ppo']:
        x_vals, y_vals, y_errs = [], [], []
        for spec in specs:
            data = other_df[(other_df['Model'] == model) & (other_df['Specification'] == spec)]
            if not data.empty:
                x_vals.append(specs.index(spec))
                y_vals.append(data['Mean Episode Length'].values[0])
                y_errs.append(data['Standard Deviation'].values[0])
        ax2.errorbar(x_vals, y_vals, yerr=y_errs, label=model_styles[model]['label'], color=model_styles[model]['color'], marker='o', linestyle='--')
    ax2.set_title('Model-Free RL Models')
    ax2.set_xticks(range(len(specs)))
    ax2.set_xticklabels([f"{x:.1e}" for x in specs])
    ax2.set_xlabel('Training Steps')
    ax2.legend()
    
    # LQR Plot
    ax3.errorbar(0, lqr_df['Mean Episode Length'].values[0], yerr=lqr_df['Standard Deviation'].values[0],
                 color='C2', marker='o', label='LQR')
    ax3.set_title('LQR Model')
    ax3.set_xticks([0])
    ax3.set_xticklabels(['LQR'])
    ax3.set_xlabel('Model')
    ax3.legend()
    
    # Save and show plot
    os.makedirs(results_folder, exist_ok=True)
    noise_str = noise_str.replace('.', '_').replace('[', '').replace(']', '')
    fig.savefig(os.path.join(results_folder, f"mean_episode_lengths_noise_{noise_str}.png"), bbox_inches="tight", dpi=300)
    plt.tight_layout()
    plt.show()
    
    return 

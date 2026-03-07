"""
Training Script for RL Agent
Uses Stable-Baselines3 to train agent on randomly generated levels
Automatically uses GPU if available via PyTorch CUDA
"""

# IMPORTANT: Set SDL to headless mode BEFORE importing pygame or environment
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Headless mode for training
os.environ['SDL_AUDIODRIVER'] = 'dummy'  # No audio needed

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from environment import PlatformerEnv
import torch
import argparse
import json
import time
import numpy as np


# Check CUDA availability
def check_cuda():
    """Check if CUDA GPU is available"""
    if torch.cuda.is_available():
        print(f"[GPU] CUDA is available!")
        print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] CUDA Version: {torch.version.cuda}")
        print(f"[GPU] Device Count: {torch.cuda.device_count()}")
        return "cuda"
    else:
        print("[CPU] CUDA not available, training will use CPU (slower)")
        return "cpu"


def create_env():
    """
    Create and wrap the environment.

    Returns:
        Wrapped environment ready for training
    """
    # Create the platformer environment
    env = PlatformerEnv(
        render_width=1920,
        render_height=1080,
        observation_width=84,
        observation_height=84,
        headless=True,  # No GUI window for training
        capture_frames=False  # Frame capture handled by callback
    )

    # Wrap with Monitor to track episode statistics (CRITICAL for reward tracking)
    env = Monitor(env, filename=None)

    # Wrap in DummyVecEnv for Stable-Baselines3 compatibility
    env = DummyVecEnv([lambda: env])

    return env


def train_agent(total_timesteps=1_000_000, save_path="models/platformer_agent"):
    """
    Train the RL agent using PPO with GPU acceleration.

    Args:
        total_timesteps: Number of training steps (1M = ~4-8 hours on GPU)
        save_path: Where to save trained model

    Returns:
        Trained model
    """
    # Check CUDA availability
    device = check_cuda()

    print(f"\n[TRAINING] Starting RL agent training")
    print(f"[TRAINING] Total timesteps: {total_timesteps:,}")
    print(f"[TRAINING] Device: {device}")
    print(f"[TRAINING] Algorithm: PPO (Proximal Policy Optimization)")

    # Create environment
    env = create_env()

    # Setup callbacks
    callbacks = setup_callbacks()

    # Create PPO model with GPU support
    # policy_kwargs can be customized for deeper networks
    policy_kwargs = dict(
        net_arch=dict(pi=[1024, 1024], vf=[1024, 1024])  # Quadrupled size for maximum learning capacity
    )

    model = PPO(
        "CnnPolicy",  # CNN policy for visual observations
        env,
        learning_rate=1e-4,  # Reduced from 3e-4 for larger model stability
        n_steps=4096,        # Increased from 2048 for better sample efficiency
        batch_size=512,      # Increased from 64 to utilize GPU and stabilize gradients
        n_epochs=8,          # Reduced from 10 to prevent overfitting with larger model
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,      # Reduced from 0.01 - larger model needs less exploration bonus
        vf_coef=0.5,
        max_grad_norm=1.0,   # Increased from 0.5 - larger model can handle larger gradients
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,  # Use CUDA if available!
        tensorboard_log="./logs/"
    )

    print(f"\n[TRAINING] Model created. Training starting...")
    print(f"[TRAINING] Hyperparameters optimized for 1024-neuron model:")
    print(f"  - Learning Rate: 1e-4 (lower for stability)")
    print(f"  - Batch Size: 512 (8x larger for GPU utilization)")
    print(f"  - Steps per Update: 4096 (2x larger for sample efficiency)")
    print(f"  - Epochs per Update: 8 (reduced to prevent overfitting)")
    print(f"  - Entropy Coefficient: 0.005 (reduced exploration bonus)")
    print(f"  - Max Gradient Norm: 1.0 (allows larger gradients)")
    print(f"\n[TRAINING] You can monitor progress in TensorBoard:")
    print(f"[TRAINING]   tensorboard --logdir ./logs/")

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10
    )

    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)

    # Update status.json to mark training as complete
    status_file = "status.json"
    try:
        # Read current status if it exists
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = json.load(f)
        else:
            status = {}

        # Mark as complete
        status['is_training'] = False
        status['completed'] = True
        status['completed_at'] = time.time()
        status['message'] = 'Training completed successfully'

        # Write updated status
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Failed to update status.json: {e}")

    print(f"\n[SUCCESS] Training complete!")
    print(f"[SUCCESS] Model saved to: {save_path}.zip")

    # Auto-export trained model to ONNX for web deployment
    print(f"\n[EXPORT] Auto-exporting model to ONNX format for web deployment...")
    try:
        import subprocess
        import sys
        export_script = os.path.join(os.path.dirname(__file__), 'export_model_onnx.py')

        # Check if export environment exists
        export_env_python = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'export_env', 'Scripts', 'python.exe')
        if os.path.exists(export_env_python):
            python_cmd = export_env_python
            print(f"[EXPORT] Using export environment: {python_cmd}")
        else:
            python_cmd = sys.executable
            print(f"[EXPORT] Using system Python: {python_cmd}")

        # Export to ONNX
        result = subprocess.run(
            [python_cmd, export_script, '--model-path', f'{save_path}.zip'],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(f"[EXPORT] Model successfully exported to ONNX!")
            print(f"[EXPORT] Output: {result.stdout}")
        else:
            print(f"[EXPORT] Warning: Model export failed: {result.stderr}")
            print(f"[EXPORT] You can manually export later using: python export_model_onnx.py")
    except Exception as e:
        print(f"[EXPORT] Warning: Auto-export failed: {e}")
        print(f"[EXPORT] You can manually export later using: python export_model_onnx.py")

    return model


class ProgressCallback(BaseCallback):
    """
    Custom callback to track and save training progress.
    Writes status.json and captures frames for intelligent visualization.
    Captures the 30th frame before death for visualization.
    """

    def __init__(self, status_file="status.json",
                 frame_dir="frames",
                 log_file="training.log",
                 save_freq=10, frame_save_freq=100,
                 verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.status_file = status_file
        self.frame_dir = frame_dir
        self.log_file = log_file
        self.save_freq = save_freq
        self.frame_save_freq = frame_save_freq

        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')

        # Frame buffer to store last 30 frames before episode ends
        self.frame_buffer = []  # Rolling buffer of last 30 frames
        self.max_buffer_size = 30

        # Episode checkpoint tracking (keep top 5 scores + last 5 recent = 10 total)
        self.episode_checkpoints = []  # List of dicts with episode info
        self.max_top_scores = 5  # Keep top 5 best scoring episodes
        self.max_recent = 5      # Keep last 5 recent episodes

        # Create directories
        status_dir = os.path.dirname(status_file)
        if status_dir:  # Only create if there's a directory component
            os.makedirs(status_dir, exist_ok=True)
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs('episode_checkpoints', exist_ok=True)

        # Clear old manifest from previous training session
        manifest_path = 'episode_checkpoints/manifest.json'
        if os.path.exists(manifest_path):
            print("[CHECKPOINT] Clearing old episode checkpoints from previous training session")
            os.remove(manifest_path)

    def _on_step(self) -> bool:
        """Called at every environment step"""

        # Capture current frame and add to rolling buffer (keeps last 30 frames)
        try:
            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                env = env.env
            current_frame = env.render(mode='rgb_array')

            # Add to buffer (rolling buffer of last 30 frames)
            self.frame_buffer.append(current_frame)
            if len(self.frame_buffer) > self.max_buffer_size:
                self.frame_buffer.pop(0)  # Remove oldest frame
        except Exception as e:
            if self.verbose > 0:
                print(f"[FRAME] Error capturing frame: {e}")

        # Update status file every N steps
        if self.num_timesteps % self.save_freq == 0:
            # Convert -inf to None for JSON serialization
            best_reward_value = None if self.best_reward == -float('inf') else float(self.best_reward)

            status = {
                'is_training': True,
                'current_step': self.num_timesteps,
                'total_steps': self.locals.get('total_timesteps', 0),
                'progress': self.num_timesteps / max(self.locals.get('total_timesteps', 1), 1),
                'episodes': len(self.episode_rewards),
                'avg_reward': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
                'best_reward': best_reward_value,
                'avg_length': float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0,
                'fps': 0,  # Would need timing to calculate
                'timestamp': time.time()
            }

            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)

        # Check for episode end
        if 'dones' in self.locals and self.locals['dones'][0]:
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0:
                info = infos[0]

                # Track episode metrics
                ep_reward = info.get('episode', {}).get('r', 0)
                ep_length = info.get('episode', {}).get('l', 0)

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Update best reward
                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward
                    # Save best model
                    self.model.save('models/platformer_agent_best')

                # Save episode checkpoint (keep last 10)
                try:
                    episode_num = len(self.episode_rewards)
                    checkpoint_path = f'episode_checkpoints/episode_{episode_num:05d}_reward_{int(ep_reward)}'

                    # Save the model
                    self.model.save(checkpoint_path)

                    # Auto-export checkpoint to ONNX for web deployment
                    try:
                        import subprocess
                        import sys
                        export_script = 'export_model_onnx.py'

                        # Check if export environment exists
                        export_env_python = os.path.join('..', 'export_env', 'Scripts', 'python.exe')
                        if os.path.exists(export_env_python):
                            python_cmd = export_env_python
                        else:
                            python_cmd = sys.executable

                        # Export this checkpoint to ONNX
                        output_dir = f'../models/episode_{episode_num}_onnx'
                        subprocess.run(
                            [python_cmd, export_script, '--model-path', f'{checkpoint_path}.zip', '--output-dir', output_dir],
                            capture_output=True,
                            timeout=60  # 1 minute timeout for checkpoint export
                        )
                        if self.verbose > 0:
                            print(f"[CHECKPOINT] Exported episode {episode_num} to ONNX")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"[CHECKPOINT] Warning: ONNX export failed for episode {episode_num}: {e}")

                    # Track checkpoint info
                    checkpoint_info = {
                        'episode': episode_num,
                        'reward': float(ep_reward),
                        'length': int(ep_length),
                        'timestep': self.num_timesteps,
                        'path': checkpoint_path,
                        'timestamp': time.time()
                    }

                    # Add to list
                    self.episode_checkpoints.append(checkpoint_info)

                    # Smart checkpoint management: Keep top 5 scores + last 5 recent
                    # This ensures we always have best models AND can see recent progress
                    import shutil

                    # Sort by reward (highest first) to get top 5 scores
                    top_scores = sorted(self.episode_checkpoints, key=lambda x: x['reward'], reverse=True)[:self.max_top_scores]
                    top_score_episodes = {cp['episode'] for cp in top_scores}

                    # Sort by episode number (most recent first) to get last 5 recent
                    recent_episodes = sorted(self.episode_checkpoints, key=lambda x: x['episode'], reverse=True)[:self.max_recent]
                    recent_episode_numbers = {cp['episode'] for cp in recent_episodes}

                    # Combine: episodes to KEEP (union of top scores and recent)
                    episodes_to_keep = top_score_episodes | recent_episode_numbers

                    # Find checkpoints to DELETE (not in keep set)
                    checkpoints_to_delete = [cp for cp in self.episode_checkpoints if cp['episode'] not in episodes_to_keep]

                    # Delete old checkpoint files
                    for old_checkpoint in checkpoints_to_delete:
                        try:
                            # Delete model file
                            if os.path.exists(old_checkpoint['path'] + '.zip'):
                                os.remove(old_checkpoint['path'] + '.zip')
                                if self.verbose > 0:
                                    print(f"[CHECKPOINT] Deleted episode {old_checkpoint['episode']} (reward: {old_checkpoint['reward']:.1f})")

                            # Delete ONNX export if exists
                            old_episode = old_checkpoint['episode']
                            old_onnx_dir = f'../models/episode_{old_episode}_onnx'
                            if os.path.exists(old_onnx_dir):
                                shutil.rmtree(old_onnx_dir)
                        except Exception as e:
                            if self.verbose > 0:
                                print(f"[CHECKPOINT] Warning: Failed to delete checkpoint {old_checkpoint['episode']}: {e}")

                    # Update checkpoint list to only keep the ones we want
                    self.episode_checkpoints = [cp for cp in self.episode_checkpoints if cp['episode'] in episodes_to_keep]

                    # Save checkpoint manifest
                    with open('episode_checkpoints/manifest.json', 'w') as f:
                        json.dump(self.episode_checkpoints, f, indent=2)

                    if self.verbose > 0:
                        print(f"[CHECKPOINT] Saved episode {episode_num} (reward: {ep_reward:.1f})")
                        # Log checkpoint summary
                        total_kept = len(self.episode_checkpoints)
                        if total_kept > 0:
                            top_rewards = sorted([cp['reward'] for cp in top_scores], reverse=True)
                            print(f"[CHECKPOINT] Keeping {total_kept} checkpoints: Top 5 scores {top_rewards[:5]} + Last 5 episodes")

                except Exception as e:
                    if self.verbose > 0:
                        print(f"[CHECKPOINT] Error saving checkpoint: {e}")

                # Intelligent frame capture (save interesting episodes)
                should_save = self._should_save_frame(
                    len(self.episode_rewards),
                    ep_reward
                )

                if should_save and len(self.frame_buffer) > 0:
                    # Save the 30th frame before death (first frame in buffer)
                    # If buffer has less than 30 frames, save the oldest available
                    try:
                        frame_filename = f"{self.frame_dir}/episode_{len(self.episode_rewards):05d}_reward_{int(ep_reward)}.png"

                        # Get the 30th frame before death (first frame in buffer)
                        frame_to_save = self.frame_buffer[0]

                        # Save the frame
                        import pygame
                        surface = pygame.surfarray.make_surface(frame_to_save.swapaxes(0, 1))
                        pygame.image.save(surface, frame_filename)

                        if self.verbose > 0:
                            frames_back = len(self.frame_buffer)
                            print(f"[FRAME] Saved {frames_back}th frame before death: {frame_filename}")

                        # Clear buffer for next episode
                        self.frame_buffer = []
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"[FRAME] Error saving frame: {e}")

        return True  # Continue training

    def _should_save_frame(self, episode_num, episode_reward):
        """
        Intelligent frame skipping logic.
        Save frames for:
        - First 10 episodes (for debugging)
        - Every 10th episode (more frequent for visibility)
        - New best reward
        - Episodes close to best reward (within 20%)
        """
        if episode_num <= 10:
            return True
        if episode_num % 10 == 0:  # Every 10th episode instead of 100th
            return True
        if episode_reward >= self.best_reward:
            return True

        # For negative rewards (early training), save if within 20% of best
        # For positive rewards, save if >= 80% of best
        if self.best_reward < 0:
            # Negative rewards: save if episode is within 20% of best (closer to 0)
            threshold = self.best_reward * 0.8  # -50 * 0.8 = -40 (better)
            if episode_reward >= threshold:
                return True
        elif self.best_reward > 0:
            # Positive rewards: save if >= 80% of best
            if episode_reward >= self.best_reward * 0.8:
                return True

        return False


def evaluate_agent(model, num_episodes=10):
    """
    Evaluate trained agent performance.

    Args:
        model: Trained model
        num_episodes: Number of test episodes

    Returns:
        dict: Evaluation metrics (avg_reward, success_rate, avg_distance)
    """
    env = create_env()

    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            done = dones[0]
            episode_reward += reward[0]
            steps += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
                if info[0].get('goal_reached', False):
                    success_count += 1

    env.close()

    return {
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'avg_length': float(np.mean(episode_lengths)),
        'success_rate': success_count / num_episodes,
        'num_episodes': num_episodes
    }


def visualize_agent(model, num_episodes=5):
    """
    Run agent and render gameplay for visualization.

    Args:
        model: Trained model
        num_episodes: Number of episodes to show
    """
    env = create_env()

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        print(f"\n[VIZ] Episode {episode + 1}/{num_episodes}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            done = dones[0]
            episode_reward += reward[0]

            if done:
                print(f"[VIZ] Reward: {episode_reward:.2f}, Steps: {info[0].get('episode_length', 0)}")
                if info[0].get('goal_reached', False):
                    print("[VIZ] Goal reached!")
                elif not info[0].get('died', True):
                    print("[VIZ] Timeout")
                else:
                    print("[VIZ] Died")

    env.close()


def setup_callbacks(save_freq=10000):
    """
    Setup training callbacks for checkpointing and logging.

    Args:
        save_freq: Save checkpoint every N steps

    Returns:
        list: Callback objects
    """
    # Progress tracking callback
    progress_callback = ProgressCallback(
        status_file="status.json",
        frame_dir="frames",
        log_file="training.log",
        save_freq=10,
        frame_save_freq=100,
        verbose=1
    )

    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='checkpoints/',
        name_prefix='checkpoint',
        verbose=1
    )

    return [progress_callback, checkpoint_callback]


if __name__ == "__main__":
    """
    Main training loop.
    Run: python train_agent.py
    Or: python train_agent.py --timesteps 500000
    """
    parser = argparse.ArgumentParser(description="Train RL platformer agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1M)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/platformer_agent",
        help="Path to save trained model"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RL PLATFORMER AGENT TRAINING")
    print("=" * 60)

    # Train the agent
    model = train_agent(
        total_timesteps=args.timesteps,
        save_path=args.save_path
    )

    # Evaluate trained model
    print("\n[EVAL] Evaluating trained model...")
    eval_results = evaluate_agent(model, num_episodes=10)

    print("\n[EVAL] Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")

    print("\n[NEXT STEPS]:")
    print("  1. Export model: python training/export_model.py")
    print("  2. Copy model to frontend: cp -r models/tfjs_model frontend/public/models/")
    print("  3. Start frontend: cd frontend && npm start")
    print("\n" + "=" * 60)

"""
Custom Gym Environment for Side-Scrolling Platformer
The AI agent receives visual input (pixels or grid) and learns to navigate randomly generated levels
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
import time

from map_generator import MapGenerator
from player import Player


class PlatformerEnv(gym.Env):
    """
    Side-scrolling platformer with procedurally generated maps.
    Agent receives visual observation of the game state.
    """

    def __init__(self, render_width=1920, render_height=1080,
                 observation_width=84, observation_height=84,
                 headless=True, capture_frames=False, reward_weights=None):
        """
        Initialize the environment.

        Args:
            render_width: Width of full game rendering (1920 for 1080p)
            render_height: Height of full game rendering (1080 for 1080p)
            observation_width: Width of observation for AI (smaller for performance)
            observation_height: Height of observation for AI
            headless: If True, use dummy video driver (faster, no window)
            capture_frames: If True, save frames for visualization
            reward_weights: Dictionary of reward weights (progress, coin, enemy, goal, death)
        """
        super(PlatformerEnv, self).__init__()

        # Set headless mode for training (no GUI window)
        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        # Initialize Pygame
        pygame.init()

        # Dimensions
        self.render_width = render_width
        self.render_height = render_height
        self.observation_width = observation_width
        self.observation_height = observation_height

        # Create surfaces: full resolution for capture, small for AI observation
        self.display_surface = pygame.Surface((render_width, render_height))
        self.observation_surface = pygame.Surface((observation_width, observation_height))

        # Observation space: 84x84x1 Grayscale image (faster training, less data)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(observation_height, observation_width, 1),
            dtype=np.uint8
        )

        # Action space: 9 discrete actions
        # 0=idle, 1=left, 2=right, 3=jump, 4=sprint+right, 5=duck
        # 6=jump+left, 7=jump+right, 8=sprint+jump+right
        self.action_space = spaces.Discrete(9)

        # Game components
        self.map_generator = MapGenerator(render_width, render_height)
        self.player = None
        self.map_data = None
        self.camera_x = 0

        # Episode tracking
        self.steps = 0
        self.max_steps = 5000  # Timeout per episode
        self.last_x = 0  # For progress reward
        self.steps_without_progress = 0  # Track steps without moving right
        self.touched_platforms = set()  # Track which platforms have been touched
        self.done = False

        # Frame capture
        self.capture_frames = capture_frames
        self.frame_buffer = []

        # Reward weights (can be configured)
        self.reward_weights = reward_weights or {
            'progress': 0.1,
            'coin': 10.0,
            'enemy': 50.0,
            'goal': 1000.0,
            'death': -100.0,
            'time': -0.01
        }

        # Initialize first episode
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset environment and generate new random map.

        Args:
            seed: Random seed for map generation
            options: Optional reset options (not used)

        Returns:
            observation: Visual observation of initial state (84x84x3 numpy array)
            info: Additional information dictionary
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Generate new random map
        self.map_data = self.map_generator.generate_map(seed=seed)

        # Create player at spawn point
        spawn = self.map_data['spawn']
        self.player = Player(spawn['x'], spawn['y'])

        # Reset tracking variables
        self.steps = 0
        self.last_x = self.player.x
        self.steps_without_progress = 0
        self.touched_platforms = set()
        self.camera_x = 0
        self.done = False

        # Clear frame buffer
        if self.capture_frames:
            self.frame_buffer = []

        # Return initial observation and info
        info = {}
        return self._get_observation(), info

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action: Integer representing action
                0 = idle
                1 = left
                2 = right
                3 = jump (straight up)
                4 = sprint + right
                5 = duck
                6 = jump + left
                7 = jump + right
                8 = sprint + jump + right

        Returns:
            observation: Visual observation after action (84x84x3)
            reward: Reward for this step
            terminated: Whether episode ended naturally (goal/death)
            truncated: Whether episode was cut off (time limit)
            info: Additional information dictionary
        """
        if self.done:
            # Episode already finished, return current state
            return self._get_observation(), 0.0, True, False, {}

        # Apply action to player
        self.player.stopMovement()

        if action == 1:  # Left
            self.player.moveLeft()
        elif action == 2:  # Right
            self.player.moveRight()
        elif action == 3:  # Jump (straight up)
            self.player.jump()
        elif action == 4:  # Sprint + Right
            self.player.sprint(True)
            self.player.moveRight()
        elif action == 5:  # Duck
            self.player.duck(True)
        elif action == 6:  # Jump + Left
            self.player.jump()
            self.player.moveLeft()
        elif action == 7:  # Jump + Right
            self.player.jump()
            self.player.moveRight()
        elif action == 8:  # Sprint + Jump + Right
            self.player.sprint(True)
            self.player.jump()
            self.player.moveRight()
        # action == 0 is idle (do nothing)

        # Update player physics
        self.player.update(self.map_data['platforms'])

        # Update camera (smooth follow)
        target_camera_x = self.player.x - 400
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

        # Check coin collection
        for coin in self.map_data['coins']:
            self.player.collectCoin(coin)

        # Update enemies and check collisions
        self._update_enemies()

        # Track episode termination
        terminated = False
        truncated = False

        # Check if player reached goal
        if self.player.checkGoalReached(self.map_data['goal']):
            terminated = True
            self.done = True

        # Check if player died
        if not self.player.isAlive:
            terminated = True
            self.done = True

        # Increment step counter
        self.steps += 1

        # Check timeout
        if self.steps >= self.max_steps:
            truncated = True
            self.done = True

        # Calculate reward
        reward = self._calculate_reward()

        # Get observation
        observation = self._get_observation()

        # Build info dictionary
        info = {
            'episode_length': self.steps,
            'distance': self.player.distance,
            'score': self.player.score,
            'goal_reached': self.player.checkGoalReached(self.map_data['goal']),
            'died': not self.player.isAlive
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        """
        Render the environment.

        Args:
            mode: 'rgb_array' for numpy array, 'human' for display (not used in headless)

        Returns:
            numpy array if mode='rgb_array', None otherwise
        """
        # Clear surface
        self.display_surface.fill((135, 206, 235))  # Sky blue

        # Draw platforms
        for platform in self.map_data['platforms']:
            # Match frontend colors exactly
            if platform['type'] == 'ground':
                color = (139, 69, 19)  # Brown - #8B4513
            elif platform['type'] == 'ice':
                color = (77, 166, 255)  # Light blue - #4DA6FF
            else:
                color = (46, 139, 87)   # Green - #2E8B57

            pygame.draw.rect(
                self.display_surface,
                color,
                (platform['x'] - self.camera_x, platform['y'],
                 platform['width'], platform['height'])
            )

        # Draw coins
        for coin in self.map_data['coins']:
            if not coin.get('collected', False):
                pygame.draw.circle(
                    self.display_surface,
                    (255, 215, 0),  # Gold
                    (int(coin['x'] - self.camera_x), int(coin['y'])),
                    coin['radius']
                )

        # Draw enemies
        for enemy in self.map_data['enemies']:
            pygame.draw.rect(
                self.display_surface,
                (255, 0, 255),  # Magenta
                (enemy['x'] - self.camera_x, enemy['y'],
                 enemy['width'], enemy['height'])
            )

        # Draw goal
        goal = self.map_data['goal']
        pygame.draw.rect(
            self.display_surface,
            (0, 255, 0),  # Green
            (goal['x'] - self.camera_x, goal['y'],
             goal['width'], goal['height'])
        )

        # Draw player
        pygame.draw.rect(
            self.display_surface,
            (255, 107, 107),  # Red - matches AI color in frontend (#FF6B6B)
            (self.player.x - self.camera_x, self.player.y,
             self.player.width, self.player.height)
        )

        if mode == 'rgb_array':
            # Convert surface to numpy array
            # Pygame uses (width, height, 3), we need (height, width, 3)
            return np.transpose(
                pygame.surfarray.array3d(self.display_surface),
                axes=(1, 0, 2)
            )

    def _get_observation(self):
        """
        Get current visual observation for the agent.
        Returns a downscaled grayscale version of the game screen.

        Returns:
            numpy array of shape (84, 84, 1) - grayscale
        """
        # Render full resolution
        rgb_array = self.render(mode='rgb_array')

        # Downscale to observation size using pygame
        # Convert numpy to surface
        temp_surface = pygame.surfarray.make_surface(
            np.transpose(rgb_array, axes=(1, 0, 2))
        )

        # Scale down
        pygame.transform.smoothscale(
            temp_surface,
            (self.observation_width, self.observation_height),
            self.observation_surface
        )

        # Convert back to numpy (RGB)
        obs_rgb = np.transpose(
            pygame.surfarray.array3d(self.observation_surface),
            axes=(1, 0, 2)
        )

        # Convert RGB to grayscale using standard luminosity formula
        # Gray = 0.299*R + 0.587*G + 0.114*B
        obs_gray = np.dot(obs_rgb[...,:3], [0.299, 0.587, 0.114])

        # Add channel dimension (84, 84) -> (84, 84, 1)
        obs_gray = np.expand_dims(obs_gray, axis=-1)

        return obs_gray.astype(np.uint8)

    def _calculate_reward(self):
        """
        Calculate reward for current state.

        Returns:
            float: Reward value
        """
        reward = 0.0

        # Progress reward (encourage moving right)
        distance_delta = self.player.x - self.last_x
        reward += distance_delta * self.reward_weights['progress']

        # Penalty for moving left
        if distance_delta < 0:
            # Player moved left, apply penalty per pixel
            reward += distance_delta * 0.5  # Negative delta * positive multiplier = negative reward

        # Track steps without rightward progress
        if distance_delta > 0:
            # Player moved right, reset counter
            self.steps_without_progress = 0
        else:
            # Player didn't move right, increment counter
            self.steps_without_progress += 1

        # Apply penalty if no progress for 50 steps
        if self.steps_without_progress >= 50:
            reward -= 100.0
            self.steps_without_progress = 0  # Reset counter after penalty

        # New platform reward (encourage exploring different platforms)
        for platform in self.map_data['platforms']:
            # Check if player is standing on this platform
            player_bottom = self.player.y + self.player.height
            platform_top = platform['y']

            # Player is on platform if:
            # - Player's feet are near platform top (within 10 pixels)
            # - Player's x position overlaps with platform
            if (abs(player_bottom - platform_top) < 10 and
                self.player.x + self.player.width > platform['x'] and
                self.player.x < platform['x'] + platform['width']):

                # Use platform position as unique ID
                platform_id = (platform['x'], platform['y'])

                # Award points if this is a new platform
                if platform_id not in self.touched_platforms:
                    self.touched_platforms.add(platform_id)
                    reward += 10.0
                break  # Only standing on one platform at a time

        self.last_x = self.player.x

        # Coin collection (score tracks this)
        # Coins add 10 to score when collected
        # We give reward based on score change

        # REMOVED: Time penalty was punishing AIs that survive longer
        # The "no progress for 50 steps" penalty handles stuck agents better

        # Goal reached (big reward)
        if self.player.checkGoalReached(self.map_data['goal']):
            reward += self.reward_weights['goal']

        # Death penalty
        if not self.player.isAlive:
            reward += self.reward_weights['death']

        return reward

    def _update_enemies(self):
        """
        Update enemy positions and check collisions.
        """
        for enemy in self.map_data['enemies']:
            # Move enemy
            enemy['x'] += enemy['direction'] * enemy['speed']

            # Simple AI: turn around at platform edges
            on_platform = None
            for platform in self.map_data['platforms']:
                if (enemy['x'] >= platform['x'] and
                    enemy['x'] <= platform['x'] + platform['width'] and
                    abs(enemy['y'] + enemy['height'] - platform['y']) < 5):
                    on_platform = platform
                    break

            if not on_platform or enemy['x'] < 0:
                enemy['direction'] *= -1

            # Check collision with player
            if self.player.checkEnemyCollision(enemy):
                self.player.die()

    def close(self):
        """
        Clean up resources.
        """
        pygame.quit()

    def save_frame(self, filename):
        """
        Save current frame as PNG for visualization.

        Args:
            filename: Path to save PNG file
        """
        pygame.image.save(self.display_surface, filename)

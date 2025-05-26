"""Manages the RL model."""
from enum import IntEnum

import numpy as np
from stable_baselines3 import DQN

class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class Tile(IntEnum):
    NO_VISION = 0
    EMPTY = 1
    RECON = 2
    MISSION = 3

class RLManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.model = DQN.load("scouts/antwerp-guard-8M/antwerp_guard_8M")

        self.size = 16

        self.obs_wall_top_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_left_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_bottom_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_right_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_rewards_space = np.full((self.size, self.size), 255, dtype=np.uint8)
        self.obs_guard_space = np.full((self.size, self.size), 128, dtype=np.uint8)

    def rl(self, observation_dict: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
        if observation_dict["scout"] == 1:
            processed_features = []
            viewcone = observation_dict.get("viewcone", [])
            for r in range(7):
                for c in range(5):
                    tile_value = viewcone[r][c] if r < len(viewcone) and c < len(viewcone[r]) else 0
                    processed_features.extend(list(np.unpackbits(np.uint8(tile_value))))
            
            direction = observation_dict.get("direction", 0)
            direction_one_hot = [0.0] * 4
            if 0 <= direction < 4: direction_one_hot[direction] = 1.0
            processed_features.extend(direction_one_hot)

            location = observation_dict.get("location", [0, 0])
            norm_x = location[0] / 16 if 16 > 0 else 0.0
            norm_y = location[1] / 16 if 16 > 0 else 0.0
            processed_features.extend([norm_x, norm_y])

            scout_role = float(observation_dict.get("scout", 0))
            processed_features.append(scout_role)

            step = observation_dict.get("step", 0)
            norm_step = step / 100 if 100 > 0 else 0.0
            processed_features.append(norm_step)
            
            # Ensure correct feature length (should be INPUT_FEATURES)
            if len(processed_features) != 288:
                # This indicates an issue with feature processing or constants
                raise ValueError(f"Feature length mismatch. Expected {288}, got {len(processed_features)}")

            processed_obs = np.array(processed_features, dtype=np.float32) # Return as numpy array for buffer

            # Your inference code goes here.
            action, _states = self.model.predict(processed_obs, deterministic=True)
            return action
        else:
            return 4

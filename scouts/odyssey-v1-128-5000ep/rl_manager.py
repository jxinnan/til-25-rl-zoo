"""Manages the RL model."""
from stable_baselines3 import DQN
import numpy as np
from collections import deque

class RLManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.model = DQN.load("scouts/odyssey-v1-128-5000ep/scout_model")
        self.obs_deque = deque(maxlen=4)  # Buffer to store recent observations

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
        
        def preprocess_features(observation):
            final_obs = []
            # one‐hot of viewcone tile type
            tile_info_map = {
                ("0", "0"): [1, 0, 0, 0],
                ("0", "1"): [0, 1, 0, 0],
                ("1", "0"): [0, 0, 1, 0],
                ("1", "1"): [0, 0, 0, 1],
            }

            def number_to_binary_string(n):
                return format(n, "08b")

            # viewcone:  shape = (viewcone_length, viewcone_width)
            for row in observation["viewcone"]:
                for tile in row:
                    b = number_to_binary_string(int(tile))
                    final_obs += tile_info_map[(b[-2], b[-1])]
                    final_obs += [int(bit) for bit in b[:6]]

            # one-hot direction (4)
            dir_oh = np.eye(4, dtype=np.float32)[observation["direction"]]
            final_obs += dir_oh.tolist()

            # one-hot location on 16×16 grid → 256
            coord_map = np.zeros((16, 16), dtype=np.float32)
            x, y = int(observation["location"][0]), int(observation["location"][1])
            coord_map[x, y] = 1.0
            final_obs += coord_map.ravel().tolist()

            final_obs.append(observation["scout"])
            final_obs.append(observation["step"])

            return final_obs
        
        if observation["scout"] == 1:
            processed_features = preprocess_features(observation)
            if len(self.obs_deque) == 0:
                # Initialize the deque with the first observation 4 times
                self.obs_deque.extend([processed_features] * 4)   
            else:
                # Append the new observation and remove the oldest one
                self.obs_deque.append(processed_features)          
            # Ensure correct feature length (should be INPUT_FEATURES)
            stacked_obs = np.array(self.obs_deque).flatten()
            if len(stacked_obs) != 612*4:
                # This indicates an issue with feature processing or constants
                raise ValueError(f"Feature length mismatch. Expected {612*4}, got {len(stacked_obs)}")

            processed_obs = np.array(stacked_obs, dtype=np.float32) # Return as numpy array for buffer

            # Your inference code goes here.
            action, _states = self.model.predict(processed_obs, deterministic=True)
            return action
        else:
            return 4

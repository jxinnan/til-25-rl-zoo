import torch
import numpy as np
from .A2CWithMapV7 import A2CWithMap


class CNNPPOPlayer:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        self.model = A2CWithMap()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(device)
        self.model.eval()
        dummy_map = np.zeros((11, 16, 16), dtype=np.float32)
        dummy_map[1, :, :] = 1.0 # Mark all tiles as unknown initially
        self.default_map = torch.from_numpy(dummy_map).float().unsqueeze(0).to(device)
        self.map_memory = self.default_map.clone()
        
    def reset(self):
        self.map_memory = self.default_map.clone()
        
    @torch.no_grad
    def __call__(self, observation: dict) -> int:
        if observation["step"] == 0:
            self.reset()
        viewcone = torch.tensor(observation["viewcone"], dtype=torch.uint8, device=self.device).unsqueeze(0)
        direction = torch.tensor(observation["direction"], dtype=torch.uint8, device=self.device).unsqueeze(0)
        location = torch.tensor(observation["location"], dtype=torch.uint8, device=self.device).unsqueeze(0)
        scout = torch.tensor(observation["scout"], dtype=torch.uint8, device=self.device).unsqueeze(0)
        step = torch.tensor(observation["step"], dtype=torch.uint8, device=self.device).unsqueeze(0)
        action_logits, _, next_map_memory = self.model(viewcone, direction, location, scout, step, self.map_memory)
        action = torch.argmax(action_logits, dim=-1).item()
        return action
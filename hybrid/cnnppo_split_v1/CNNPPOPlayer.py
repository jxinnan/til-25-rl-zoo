import torch
import numpy as np
from .A2CWithMapV7 import A2CWithMap
from typing import Union

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
        self.map_memory = next_map_memory
        action = torch.argmax(action_logits, dim=-1).item()
        return action
    
class CNNPPOPlayerSplit:
    def __init__(self, model_ckpt: str, device: Union[str, torch.device] = "cuda:0"):
        self.device = device
        checkpoint = torch.load(model_ckpt, map_location=self.device)
        self.scout_model = A2CWithMap().to(self.device)
        self.guard_model = A2CWithMap().to(self.device)
        self.scout_model.load_state_dict(checkpoint['model_scout_state_dict'])
        self.guard_model.load_state_dict(checkpoint['model_guard_state_dict'])
        self.scout_model.eval()
        self.guard_model.eval()
        dummy_map = np.zeros((11, 16, 16), dtype=np.float32)
        dummy_map[1, :, :] = 1.0 # Mark all tiles as unknown initially
        self.default_map = torch.from_numpy(dummy_map).float().unsqueeze(0).to(device)
        self.reset(0)
    
    def reset(self, scout: int):
        self.map_memory = self.default_map.clone()
        if scout:
            self.model = self.scout_model
        else:
            self.model = self.guard_model
    
    def preprocess_observation(self, observation: dict) -> tuple:
        viewcone = torch.tensor(observation["viewcone"], dtype=torch.uint8, device=self.device).unsqueeze(0)
        direction = torch.tensor(observation["direction"], dtype=torch.uint8, device=self.device).unsqueeze(0)
        location = torch.tensor(observation["location"], dtype=torch.uint8, device=self.device).unsqueeze(0)
        scout = torch.tensor(observation["scout"], dtype=torch.uint8, device=self.device).unsqueeze(0)
        step = torch.tensor(observation["step"], dtype=torch.uint8, device=self.device).unsqueeze(0)
        return viewcone, direction, location, scout, step
    
    @torch.no_grad
    def __call__(self, observation: dict) -> int:
        if observation["step"] == 0:
            self.reset(observation["scout"])
        preprocessed_observation = self.preprocess_observation(observation)
        action_logits, _, next_map_memory = self.model(*preprocessed_observation, self.map_memory)
        self.map_memory = next_map_memory
        action = torch.argmax(action_logits, dim=-1).item()
        return action
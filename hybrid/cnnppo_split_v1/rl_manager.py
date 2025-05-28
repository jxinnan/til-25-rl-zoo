import torch

from .CNNPPOPlayer import CNNPPOPlayerSplit

class RLManager:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNPPOPlayerSplit("hybrid/cnnppo_split_v1/checkpoint_episode_18800_v1.pth", device)

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        return self.model(observation)
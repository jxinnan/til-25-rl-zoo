import torch

from .CNNPPOPlayer import CNNPPOPlayer

class RLManager:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNPPOPlayer("hybrid/cnnppo_v7e7/v7e7_ep1200.pth", device)

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        return self.model(observation)
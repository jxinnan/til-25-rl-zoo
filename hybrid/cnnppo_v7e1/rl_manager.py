from .CNNPPOPlayer import CNNPPOPlayer

class RLManager:
    def __init__(self):
        self.model = CNNPPOPlayer("hybrid/cnnppo_v7e1/v7e1_ep31600.pth")

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        return self.model(observation)
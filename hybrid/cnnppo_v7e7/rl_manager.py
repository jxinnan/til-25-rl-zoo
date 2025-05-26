from .CNNPPOPlayer import CNNPPOPlayer

class RLManager:
    def __init__(self):
        self.model = CNNPPOPlayer("hybrid/cnnppo_v7e7/v7e7_ep1200.pth")

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        return self.model(observation)
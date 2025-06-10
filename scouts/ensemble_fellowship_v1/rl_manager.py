from importlib import import_module

import numpy as np

class RLManager:
    def __init__(self):
        voter_register = [
            "atlanta-8M-deadend-8M",
            "avignon-8M",
            "avignon-ariane4-8M",
            "avignon-ariane20-8M",
            "avignon-ariane24-4M",
            "avignon-ariane24-normal-2M4",
            "avignon-ariane25-4M",
            "avignon-ariane25-normal-2M8",
            "caracas-8M",
            "caracas-ariane25-4M",
            "chitose-6M",
            "chitose-8M",
        ]
        weights = [0, 0.1758735867204316, 0, 0, 0, 0.5744209722209784, 0, 0, 0, 0, 0, 0,]
        assert len(voter_register) == len(weights)

        i = 0
        while i < len(weights):
            if weights[i] == 0:
                voter_register.pop(i)
                weights.pop(i)
            else:
                i += 1

        self.voters = [getattr(import_module(f"ensemble_scouts.{voter_name}.rl_manager"), "RLManager")() for voter_name in voter_register]
        self.weights = weights

        self.action = 4

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        ballot_box = [weight * voter.rl(observation, self.action) for voter, weight in zip(self.voters, self.weights)]
        
        election_results = np.zeros((4,))
        for vote in ballot_box:
            assert len(vote) == 4
            election_results += vote
        
        self.action = np.argmax(election_results)

        return self.action
    
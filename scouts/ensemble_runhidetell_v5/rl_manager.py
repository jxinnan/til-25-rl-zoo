from importlib import import_module

import numpy as np

class RLManager:
    def __init__(self):
        voter_register = [
            "chitose-guard2-8M",
            "chitose-guard-2M",
            "chitose-guard-8M",
            "cologne-guard-8M",
        ]
        weights = [0.53438876791812, 0.16015668791131196, 0.2781693609504954, 0.02728518322007276]
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
    
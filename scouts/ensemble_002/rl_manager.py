from importlib import import_module

import torch

class RLManager:
    def __init__(self):
        voter_register = [
            "avignon-ariane20-4M-backAndForth",
            # "avignon-ariane24-normal-2M4",
            # "avignon-ariane25-4M",
            # "avignon-ariane20-8M",
        ]
        with torch.device('cpu'):
            self.voters = [getattr(import_module(f"ensemble_scouts.{voter_name}.rl_manager"), "RLManager")() for voter_name in voter_register]
    
    def rl(self, observation: dict[str, int | list[int]]) -> int:
        with torch.device('cpu'):
            election_results = [int(voter.rl(observation)) for voter in self.voters]

        return max(set(election_results), key=election_results.count)
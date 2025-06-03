from importlib import import_module

import torch

class RLManager:
    def __init__(self):
        voter_register = [
            "avignon-8M",
            "avignon-ariane5-8M",
            "avignon-ariane7-8M",
            "avignon-ariane8-8M",
            "avignon-ariane10-4M",
            "avignon-ariane12-4M",
            "avignon-ariane13-4M",
            "avignon-ariane17-4M",
            "avignon-ariane19-4M",
        ]
        with torch.device('cpu'):
            self.voters = [getattr(import_module(f"scouts.{voter_name}.rl_manager"), "RLManager")() for voter_name in voter_register]
    
    def rl(self, observation: dict[str, int | list[int]]) -> int:
        with torch.device('cpu'):
            election_results = [int(voter.rl(observation)) for voter in self.voters]

        return max(set(election_results), key=election_results.count)
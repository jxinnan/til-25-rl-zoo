from random import randint

class RLManager:
    def rl(self, observation: dict[str, int | list[int]]) -> int:
        return randint(0,3)
import numpy as np

from enum import IntEnum


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


class State:
    def __init__(self, loc: tuple[int, int], dir: int, action: int) -> None:
        self.loc = (int(loc[0]), int(loc[1]))
        self.dir = int(dir)
        self.action = int(action)

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        
        return self.loc[0] == other.loc[0] and self.loc[1] == other.loc[1] and self.dir == other.dir
 

class RLManager:
    SCOUT_REWARDS = {
        'empty': 0,
        'hidden': 1.2,
        'recon': 1,
        'mission': 5,
        'guard': -70,

        # Repeated Penalty
        'repeat': -0.3,

        # Wall Penalty
        'wall': -0.1, 

        # Backwards Penalty
        'backwards': -0.2
    }
    SCOUT_MAX_DEPTH = 8

    GUARD_REWARDS = {
        'empty': 0,
        'hidden': 1.2,
        'recon': 0,
        'mission': 0,
        'scout': 50,
        
        # Encourages guards to top left before T30, and discourages them afterwards
        'target': (4, 4),
        'preferential_turns': 30,
        'preferential_drive': -0.2,
        # 'dead_end_penalty': -1,
        
        # Repeated Penalty
        'repeat': -0.3,

        # Wall Penalty
        'wall': -0.2,

        # Backwards Penalty
        'backwards': -0.4
    }
    GUARD_MAX_DEPTH = 7

    DELTA = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self):
        self.size = 16

        # Loaded in `load_rewards`
        self.rewards = None
        self.max_depth = None
        self.value = None

        # Number of times visited
        self.num_visited = np.zeros((self.size, self.size), dtype=np.uint8)

        # Tracker
        self.step = 0

        # Wall maps
        self.obs_wall = np.zeros((4, self.size, self.size), dtype=np.uint8)

        # Set walls
        self.obs_wall[0, self.size-1, :] = np.uint8(1)  # RIGHT
        self.obs_wall[1, :, self.size-1] = np.uint8(1)  # BOTTOM
        self.obs_wall[2, 0, :] = np.uint8(1)            # LEFT
        self.obs_wall[3, :, 0] = np.uint8(1)            # TOP

        self.curr_guard_pos = []

        # For Scouts
        self.guards_tracker : dict[tuple[int, int], float] = {}
        self.decay_factor = self.update_decay_factor()

        # For Guards 
        self.scout_pos : tuple[int, int] | None = None
        self.decay_multiplier = 1

    def load_rewards(self):
        if self.is_scout:
            self.rewards = self.SCOUT_REWARDS
            self.max_depth = self.SCOUT_MAX_DEPTH
        else:
            self.rewards = self.GUARD_REWARDS
            self.max_depth = self.GUARD_MAX_DEPTH
        
        self.value = np.full((self.size, self.size), self.rewards['hidden'])
    
    def print_arr(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                print(arr[j][i], end = " ")

            print()
        print()

    def update_decay_factor(self, num_guards_spotted: int = 0) -> None:
        self.decay_factor = (1 - (1 + 3*num_guards_spotted)/10)

    def update_map(self, viewcone: list[list[int]], location: tuple[int, int], direction: int) -> None:
        new_grid = np.array(viewcone, dtype=np.uint8)
        curr_direction = np.array(direction, dtype=np.int8) # right down left up
        curr_location = np.array(location, dtype=np.int8)

        self.num_visited[curr_location[0]][curr_location[1]] += 1

        # rotate clockwise so absolute north faces up
        new_grid = np.rot90(new_grid, k=curr_direction)

        match curr_direction: # location of self in rotated new_gridview
            case Direction.RIGHT: 
                rel_curr_location = (2,2)
            case Direction.DOWN: 
                rel_curr_location = (2,2)
            case Direction.LEFT: 
                rel_curr_location = (4,2) 
            case Direction.UP: 
                rel_curr_location = (2,4)

        r, c = new_grid.shape

        self.curr_guard_pos = []

        # Reset `scout_pos`for guard
        if not self.is_scout and self.scout_pos:
            if curr_location[0] == self.scout_pos[0] and curr_location[1] == self.scout_pos[1]:
                self.scout_pos = None
                self.decay_multiplier = 1
            else:
                self.decay_multiplier /= 2

        for i in range(r):
            true_i = curr_location[0] + (i - rel_curr_location[0])
            if true_i > 15 or true_i < 0: 
                continue 

            for j in range(c):
                true_j = curr_location[1] + (j - rel_curr_location[1])
                if true_j > 15 or true_j < 0: 
                    continue

                tile = new_grid[i][j]
                last2 = tile & 0b11
                match last2:
                    case 0b01:
                        self.value[true_i][true_j] = self.rewards['empty']
                    case 0b10:
                        self.value[true_i][true_j] = self.rewards['recon']
                    case 0b11:
                        self.value[true_i][true_j] = self.rewards['mission']
                    case _:
                        pass    

                walls = [tile & 0b10000, tile & 0b100000, tile & 0b1000000, tile & 0b10000000]

                # WHY DOES REPLACING `curr_direction` with `4-curr_direction` work???
                # walls = walls[curr_direction:] + walls[:curr_direction]
                walls = walls[4-curr_direction:] + walls[:4-curr_direction]

                for wall_dir in range(len(walls)):
                    if not walls[wall_dir]: 
                        continue

                    self.obs_wall[wall_dir][true_i][true_j] = 1

                    opp_i, opp_j = true_i + self.DELTA[wall_dir][0], true_j + self.DELTA[wall_dir][1]
                    if opp_i >= self.size or opp_i < 0 or opp_j >= self.size or opp_j < 0:
                        continue
                        
                    self.obs_wall[(wall_dir+2)%4][opp_i][opp_j] = 1

                if tile & 0b100:
                    self.scout_pos = (true_i, true_j)
                    self.decay_multiplier = 1
                if tile & 0b1000:
                    self.curr_guard_pos.append((true_i, true_j))
                if (true_i, true_j) in self.guards_tracker:
                    del self.guards_tracker[(true_i, true_j)]

        if not self.is_scout:
            return
        
        self.update_decay_factor(len(self.curr_guard_pos))
        if self.decay_factor:
            for k in self.guards_tracker:
                self.guards_tracker[k] *= self.decay_factor
        else: 
            self.guards_tracker.clear()
        
        for pos in self.curr_guard_pos:
            self.guards_tracker[pos] = self.rewards["guard"]

    def get_num_repeats(self, path: list[State], curr_loc: tuple[int, int]) -> int:
        return sum(curr_loc == path[idx].loc for idx in range(len(path)-1))
                                         
    def score_tile_scout(self, path: list[State]) -> float:
        i, j = path[-1].loc
        
        num_repeats = self.get_num_repeats(path, (i, j))

        action_score = self.value[i][j] if not num_repeats else 0
        action_score += (
            (self.num_visited[i][j] + num_repeats) * self.rewards["repeat"] + 
            sum(dir_walls[i][j] for dir_walls in self.obs_wall) * self.rewards["wall"] + 
            self.guards_tracker.get((i, j), 0)
        )
        
        return action_score
    
    def score_tile_guard(self, path: list[State]) -> float:
        i, j = path[-1].loc
        action_score = self.rewards["scout"] * self.decay_multiplier if self.scout_pos == (i, j) else 0
        
        manhattan_dist = abs(i - self.rewards["target"][0]) + abs(j - self.rewards["target"][1])

        if self.step < self.rewards["preferential_turns"]:
            action_score += self.rewards["preferential_drive"] * manhattan_dist

        num_repeats = self.get_num_repeats(path, (i, j))
        action_score += (self.num_visited[i][j] + num_repeats) * self.rewards['repeat']

        return action_score

    def recur(self, path: list[State]) -> float | int: 
        if len(path)-1 == min(self.max_depth, 100 - self.step):
            return 0
        
        i, j = path[-1].loc
        dir = path[-1].dir

        prev_state = path[-2] if len(path) > 1 else None

        new_states = []

        # FORWARD BACKWARD MOVEMENT
        opp_dir = (dir+2)%4
        if not self.obs_wall[dir][i][j]:
            new_i, new_j = i + self.DELTA[dir][0], j + self.DELTA[dir][1]
            if self.is_scout or (new_i, new_j) not in self.curr_guard_pos:
                new_states.append(State((new_i, new_j), dir, 0)) # Forward Movement
        if not self.obs_wall[opp_dir][i][j]:
            new_i, new_j = i + self.DELTA[opp_dir][0], j + self.DELTA[opp_dir][1]
            if self.is_scout or (new_i, new_j) not in self.curr_guard_pos:
                new_states.append(State((new_i, new_j), dir, 1)) # Backward Movement

        # TURNING LEFT RIGHT
        turn_left = State((i, j), (dir+3)%4, 2)
        turn_right = State((i, j), (dir+1)%4, 3)

        if turn_left != prev_state:
            new_states.append(turn_left)
        if turn_right != prev_state:
            new_states.append(turn_right)

        scores = [] 
        for new_state in new_states:
            updated_path = path + [new_state]
            action_score = self.score_tile_scout(updated_path) if self.is_scout else self.score_tile_guard(updated_path)
            action_score += self.recur(updated_path)

            if new_state.action == 1:
                action_score += self.rewards['backwards']

            scores.append(action_score)

        if len(path) > 1:
            return sum(scores)/len(scores)
        
        best_move = max(zip(scores, new_states), key=lambda x: x[0])[1]

        return best_move.action

    def rl(self, observation_json: dict) -> int:
        viewcone = observation_json["viewcone"]
        direction = np.array(observation_json["direction"], np.uint8)
        location = np.array(observation_json["location"], np.uint8)
        self.step = observation_json["step"]

        if self.step == 0: 
            self.is_scout = bool(observation_json["scout"])
            self.load_rewards()

        self.update_map(viewcone, location, direction)

        state_hist = [State((location[0], location[1]), direction, 4)]

        action = self.recur(state_hist)

        return action

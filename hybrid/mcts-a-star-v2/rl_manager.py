import numpy as np
import math
import random
import heapq
from enum import IntEnum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


@dataclass
class State:
    loc: Tuple[int, int]
    dir: int
    action: int
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.loc[0] == other.loc[0] and self.loc[1] == other.loc[1] and self.dir == other.dir


class MCTSNode:
    def __init__(self, state: Tuple[int, int], parent=None, action=None):
        self.state = state  # (x, y) position
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]
    
    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)


class MCTSGoalPlanner:
    def __init__(self, rl_manager):
        self.rl_manager = rl_manager
        self.size = rl_manager.size
        self.DELTA = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.simulation_depth = 15
        self.exploration_constant = 1.4
        
    def get_valid_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions considering walls and bounds"""
        # Ensure we're working with regular Python integers
        x, y = int(pos[0]), int(pos[1])
        neighbors = []
        
        for i, (dx, dy) in enumerate(self.DELTA):
            new_x, new_y = int(x + dx), int(y + dy)
            
            # Check bounds
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                # Check if there's a wall blocking this direction
                if not self.rl_manager.obs_wall[i][x][y]:
                    # For guards, avoid positions with other guards
                    if not self.rl_manager.is_scout:
                        if (new_x, new_y) not in self.rl_manager.curr_guard_pos:
                            neighbors.append((new_x, new_y))
                    else:
                        neighbors.append((new_x, new_y))
        
        return neighbors
    
    def evaluate_position(self, pos: Tuple[int, int]) -> float:
        """Evaluate the reward value of a position"""
        # Ensure we're working with regular Python integers
        x, y = int(pos[0]), int(pos[1])
        base_reward = self.rl_manager.value[x][y]
        
        if self.rl_manager.is_scout:
            # Scout evaluation
            guard_penalty = self.rl_manager.guards_tracker.get(pos, 0)
            repeat_penalty = self.rl_manager.num_visited[x][y] * self.rl_manager.rewards['repeat']
            wall_penalty = sum(dir_walls[x][y] for dir_walls in self.rl_manager.obs_wall) * self.rl_manager.rewards['wall']
            
            return base_reward + guard_penalty + repeat_penalty + wall_penalty
        else:
            # Guard evaluation
            scout_reward = 0
            if self.rl_manager.scout_pos == pos:
                scout_reward = self.rl_manager.rewards['scout'] * self.rl_manager.decay_multiplier
            
            # Manhattan distance to target for preferential movement
            manhattan_dist = abs(x - self.rl_manager.rewards['target'][0]) + abs(y - self.rl_manager.rewards['target'][1])
            preferential_bonus = 0
            if self.rl_manager.step < self.rl_manager.rewards['preferential_turns']:
                preferential_bonus = self.rl_manager.rewards['preferential_drive'] * manhattan_dist
            
            repeat_penalty = self.rl_manager.num_visited[x][y] * self.rl_manager.rewards['repeat']
            
            return scout_reward + preferential_bonus + repeat_penalty
    
    def simulate(self, start_pos: Tuple[int, int]) -> float:
        """Monte Carlo simulation from a given position"""
        current_pos = start_pos
        total_reward = 0.0
        
        for _ in range(self.simulation_depth):
            neighbors = self.get_valid_neighbors(current_pos)
            if not neighbors:
                break
                
            # Random rollout with slight bias towards higher rewards
            neighbor_rewards = [self.evaluate_position(pos) for pos in neighbors]
            
            # Softmax selection with temperature
            temperature = 0.5
            exp_rewards = np.exp(np.array(neighbor_rewards) / temperature)
            probabilities = exp_rewards / np.sum(exp_rewards)
            
            next_pos = np.random.choice(len(neighbors), p=probabilities)
            current_pos = neighbors[next_pos]
            
            reward = self.evaluate_position(current_pos)
            total_reward += reward * (0.9 ** _)  # Discount factor
        
        return total_reward
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCT"""
        while not node.untried_actions and node.children:
            node = node.best_child(self.exploration_constant)
        return node
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand the tree by adding a new child node"""
        if node.untried_actions:
            action = node.untried_actions.pop()
            new_state = action
            child_node = MCTSNode(new_state, parent=node, action=action)
            node.children.append(child_node)
            return child_node
        return node
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate the reward up the tree"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def mcts_search(self, start_pos: Tuple[int, int], iterations: int = 1000) -> Tuple[int, int]:
        """Main MCTS search to find the best goal position"""
        root = MCTSNode(start_pos)
        root.untried_actions = self.get_valid_neighbors(start_pos)
        
        for _ in range(iterations):
            # Selection
            selected_node = self.select(root)
            
            # Expansion
            if selected_node.untried_actions:
                selected_node = self.expand(selected_node)
                # Initialize untried actions for new node
                selected_node.untried_actions = self.get_valid_neighbors(selected_node.state)
            
            # Simulation
            reward = self.simulate(selected_node.state)
            
            # Backpropagation
            self.backpropagate(selected_node, reward)
        
        # Return the best child's state as the goal
        if root.children:
            best_child = max(root.children, key=lambda c: c.value / c.visits if c.visits > 0 else 0)
            return best_child.state
        else:
            return start_pos


class AStarPathPlanner:
    def __init__(self, rl_manager):
        self.rl_manager = rl_manager
        self.size = rl_manager.size
        self.DELTA = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], direction: int) -> float:
        """Calculate the cost of moving from one position to another"""
        # Ensure we're working with regular Python integers
        x, y = int(to_pos[0]), int(to_pos[1])
        from_x, from_y = int(from_pos[0]), int(from_pos[1])
        
        # Base movement cost
        base_cost = 1.0
        
        # Check for walls
        if self.rl_manager.obs_wall[direction][from_x][from_y]:
            return float('inf')  # Can't move through walls
        
        # Dynamic guard avoidance
        guard_penalty = 0.0
        if not self.rl_manager.is_scout:
            # Guards avoid other guards
            if to_pos in self.rl_manager.curr_guard_pos:
                return float('inf')
        else:
            # Scouts get penalty for being near guards
            for guard_pos in self.rl_manager.curr_guard_pos:
                dist_to_guard = self.manhattan_distance(to_pos, guard_pos)
                if dist_to_guard <= 2:  # Guard detection range
                    guard_penalty += (3 - dist_to_guard) * 5.0
        
        # Reward bonus for high-value tiles (negative cost = positive reward)
        reward_bonus = -self.rl_manager.value[x][y] * 0.1
        
        # Penalty for revisiting
        revisit_penalty = self.rl_manager.num_visited[x][y] * 0.5
        
        return base_cost + guard_penalty + reward_bonus + revisit_penalty
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int, float]]:
        """Get valid neighbors with their direction and movement cost"""
        # Ensure we're working with regular Python integers
        x, y = int(pos[0]), int(pos[1])
        neighbors = []
        
        for i, (dx, dy) in enumerate(self.DELTA):
            new_x, new_y = int(x + dx), int(y + dy)
            
            # Check bounds
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                cost = self.get_movement_cost(pos, (new_x, new_y), i)
                if cost != float('inf'):
                    neighbors.append(((new_x, new_y), i, cost))
        
        return neighbors
    
    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to goal"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding with dynamic guard avoidance and reward prioritization"""
        if start == goal:
            return [start]
        
        # Priority queue: (f_score, position)
        open_set = [(0, start)]
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                return self.reconstruct_path(came_from, current)
            
            for neighbor, direction, move_cost in self.get_neighbors(current):
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    # Heuristic: Manhattan distance + reward potential
                    h_score = self.manhattan_distance(neighbor, goal)
                    
                    # Add reward potential to heuristic (encourage high-reward paths)
                    if self.rl_manager.value[neighbor[0]][neighbor[1]] > 0:
                        h_score -= self.rl_manager.value[neighbor[0]][neighbor[1]] * 0.1
                    
                    f_score[neighbor] = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return []


class MCTSAStarManager:
    def __init__(self, rl_manager):
        self.rl_manager = rl_manager
        self.mcts_planner = MCTSGoalPlanner(rl_manager)
        self.astar_planner = AStarPathPlanner(rl_manager)
        self.current_goal = None
        self.current_path = []
        self.path_index = 0
        self.replan_frequency = 5  # Replan every N steps
        self.last_replan_step = 0
        
    def should_replan(self) -> bool:
        """Determine if we should replan based on various conditions"""
        # Replan if we've reached our goal
        if not self.current_path or self.path_index >= len(self.current_path):
            return True
        
        # Replan periodically
        if self.rl_manager.step - self.last_replan_step >= self.replan_frequency:
            return True
        
        # Replan if guards have moved significantly (for scouts)
        if self.rl_manager.is_scout and len(self.rl_manager.curr_guard_pos) > 0:
            return True
        
        # Replan if scout position changed (for guards)
        if not self.rl_manager.is_scout and self.rl_manager.scout_pos:
            return True
        
        return False
    
    def plan_and_execute(self, current_pos: Tuple[int, int], current_dir: int) -> int:
        """Main planning function that combines MCTS goal selection with A* pathfinding"""
        
        # Ensure we're working with regular Python integers
        current_pos = (int(current_pos[0]), int(current_pos[1]))
        current_dir = int(current_dir)
        
        # Check if we need to replan
        if self.should_replan():
            # Use MCTS to find the best goal position
            self.current_goal = self.mcts_planner.mcts_search(current_pos, iterations=500)
            
            # Use A* to find path to goal
            self.current_path = self.astar_planner.a_star(current_pos, self.current_goal)
            self.path_index = 1  # Skip current position
            self.last_replan_step = self.rl_manager.step
        
        # Execute current path
        if self.current_path and self.path_index < len(self.current_path):
            next_pos = self.current_path[self.path_index]
            
            # Calculate the action needed to reach next position
            action = self.get_action_to_position(current_pos, next_pos, current_dir)
            
            # Move to next position in path if we're moving forward
            if action == 0:  # Forward movement
                self.path_index += 1
            
            return action
        
        # Fallback: stay in place or turn
        return 2  # Turn left
    
    def get_action_to_position(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], current_dir: int) -> int:
        """Convert a target position to an action (0=forward, 1=backward, 2=turn_left, 3=turn_right)"""
        # Ensure we're working with regular Python integers
        current_pos = (int(current_pos[0]), int(current_pos[1]))
        target_pos = (int(target_pos[0]), int(target_pos[1]))
        current_dir = int(current_dir)
        
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Determine required direction
        if dx == 1 and dy == 0:
            required_dir = Direction.RIGHT
        elif dx == -1 and dy == 0:
            required_dir = Direction.LEFT
        elif dx == 0 and dy == 1:
            required_dir = Direction.DOWN
        elif dx == 0 and dy == -1:
            required_dir = Direction.UP
        else:
            # Target is not adjacent or is current position
            return 2  # Default to turn left
        
        # Calculate action needed
        if current_dir == required_dir:
            return 0  # Forward
        elif (current_dir + 2) % 4 == required_dir:
            return 1  # Backward
        elif (current_dir + 1) % 4 == required_dir:
            return 3  # Turn right
        elif (current_dir + 3) % 4 == required_dir:
            return 2  # Turn left
        else:
            return 2  # Default to turn left

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
    SCOUT_MAX_DEPTH = 7

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
        self.mcts_astar_manager = None

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
        # Call original update logic
        viewcone = observation_json["viewcone"]
        direction = np.array(observation_json["direction"], np.uint8)
        location = np.array(observation_json["location"], np.uint8)
        self.step = observation_json["step"]

        if self.step == 0: 
            self.is_scout = bool(observation_json["scout"])
            self.load_rewards()
            # Initialize MCTS+A* manager
            self.mcts_astar_manager = MCTSAStarManager(self)

        self.update_map(viewcone, location, direction)
        
        # Use MCTS+A* planning instead of original recursive approach
        if self.mcts_astar_manager:
            return self.mcts_astar_manager.plan_and_execute(
                (location[0], location[1]), direction
            )
        else:
            # Fallback to original method
            state_hist = [State((location[0], location[1]), direction, 4)]
            return self.recur(state_hist)

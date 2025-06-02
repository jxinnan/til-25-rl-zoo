import numpy as np
import math
import heapq
from enum import IntEnum
from typing import List, Tuple, Optional, Dict
import random


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


class MCTSNode:
    def __init__(self, location: Tuple[int, int], parent=None, action=None):
        self.location = location
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = []
        self.is_terminal = False
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=1.4):
        """Select child using UCT formula"""
        if not self.children:
            return None
            
        choices_weights = [
            (child.total_reward / max(child.visits, 1)) + c_param * math.sqrt((2 * math.log(self.visits) / max(child.visits, 1)))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]
    
    def add_child(self, location: Tuple[int, int], action):
        child = MCTSNode(location, parent=self, action=action)
        self.untried_actions.remove(action)
        self.children.append(child)
        return child


class RLManager:
    SCOUT_REWARDS = {
        'empty': 0,
        'hidden': 1.2,
        'recon': 1,
        'mission': 5,
        'guard': -70,
        'repeat': -0.3,
        'wall': -0.1,
        'backwards': -0.2
    }
    SCOUT_MAX_DEPTH = 7

    GUARD_REWARDS = {
        'empty': 0,
        'hidden': 1.2,
        'recon': 0,
        'mission': 0,
        'scout': 50,
        'target': (4, 4),
        'preferential_turns': 30,
        'preferential_drive': -0.2,
        'repeat': -0.3,
        'wall': -0.2,
        'backwards': -0.4
    }
    GUARD_MAX_DEPTH = 7

    DELTA = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self):
        self.size = 16
        self.rewards = None
        self.max_depth = None
        self.value = None
        self.num_visited = np.zeros((self.size, self.size), dtype=np.uint8)
        self.step = 0
        self.obs_wall = np.zeros((4, self.size, self.size), dtype=np.uint8)
        
        # Set boundary walls
        self.obs_wall[0, self.size-1, :] = np.uint8(1)  # RIGHT
        self.obs_wall[1, :, self.size-1] = np.uint8(1)  # BOTTOM
        self.obs_wall[2, 0, :] = np.uint8(1)            # LEFT
        self.obs_wall[3, :, 0] = np.uint8(1)            # TOP

        self.curr_guard_pos = []
        self.guards_tracker: Dict[Tuple[int, int], float] = {}
        self.decay_factor = 0.7
        self.scout_pos: Optional[Tuple[int, int]] = None
        self.decay_multiplier = 1
        
        # MCTS specific attributes
        self.current_goal: Optional[Tuple[int, int]] = None
        self.path_to_goal: List[Tuple[int, int]] = []
        self.mcts_iterations = 50
        self.goal_search_radius = 8

    def load_rewards(self):
        if self.is_scout:
            self.rewards = self.SCOUT_REWARDS
            self.max_depth = self.SCOUT_MAX_DEPTH
        else:
            self.rewards = self.GUARD_REWARDS
            self.max_depth = self.GUARD_MAX_DEPTH
        
        self.value = np.full((self.size, self.size), self.rewards['hidden'])

    def update_decay_factor(self, num_guards_spotted: int = 0) -> None:
        self.decay_factor = (1 - (1 + 3*num_guards_spotted)/10)

    def update_map(self, viewcone: List[List[int]], location: Tuple[int, int], direction: int) -> None:
        new_grid = np.array(viewcone, dtype=np.uint8)
        curr_direction = np.array(direction, dtype=np.int8)
        curr_location = np.array(location, dtype=np.int8)

        self.num_visited[curr_location[0]][curr_location[1]] += 1

        # Rotate grid to absolute orientation
        new_grid = np.rot90(new_grid, k=curr_direction)

        match curr_direction:
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

        # Reset scout_pos for guard
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

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds"""
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size

    def can_move_between(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Check if movement between two adjacent positions is possible"""
        if not self.is_valid_position(to_pos):
            return False
            
        # Calculate direction of movement
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        
        if abs(dx) + abs(dy) != 1:  # Not adjacent
            return False
            
        # Find direction index
        for dir_idx, (ddx, ddy) in enumerate(self.DELTA):
            if dx == ddx and dy == ddy:
                return not self.obs_wall[dir_idx][from_pos[0]][from_pos[1]]
        
        return False

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        neighbors = []
        for dx, dy in self.DELTA:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self.can_move_between(pos, new_pos):
                neighbors.append(new_pos)
        return neighbors

    def can_reach_position(self, start: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """Check if target position can be reached from start using simple BFS"""
        if start == target:
            return True
            
        visited = set()
        queue = [start]
        max_steps = 20  # Limit search to prevent infinite loops
        
        while queue and max_steps > 0:
            current = queue.pop(0)
            if current == target:
                return True
                
            if current in visited:
                continue
                
            visited.add(current)
            max_steps -= 1
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return False

    def find_exploration_targets(self, center: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find positions that lead to unexplored areas"""
        exploration_targets = []
        
        # Look for positions adjacent to unexplored (hidden) areas
        for i in range(self.size):
            for j in range(self.size):
                if self.value[i][j] == self.rewards['hidden']:
                    # Find accessible positions near this unexplored area
                    for di, dj in self.DELTA:
                        adjacent_pos = (i + di, j + dj)
                        if (self.is_valid_position(adjacent_pos) and 
                            self.value[adjacent_pos[0]][adjacent_pos[1]] != self.rewards['hidden'] and
                            self.can_reach_position(center, adjacent_pos)):
                            # Give higher reward for positions that border unexplored areas
                            reward = 2.0 - self.manhattan_distance(center, adjacent_pos) * 0.1
                            exploration_targets.append((adjacent_pos, reward))
        
        # If no exploration targets, just find any reachable position
        if not exploration_targets:
            for i in range(self.size):
                for j in range(self.size):
                    pos = (i, j)
                    if (pos != center and 
                        self.is_valid_position(pos) and 
                        self.can_reach_position(center, pos)):
                        # Prefer less visited positions
                        reward = 1.0 - self.num_visited[i][j] * 0.1 - self.manhattan_distance(center, pos) * 0.05
                        exploration_targets.append((pos, reward))
        
        return exploration_targets

    def is_facing_wall(self, pos: Tuple[int, int], direction: int) -> bool:
        """Check if agent is facing a wall in the given direction"""
        return self.obs_wall[direction][pos[0]][pos[1]] == 1

    def get_alternative_directions(self, pos: Tuple[int, int], current_dir: int) -> List[int]:
        """Get alternative directions when facing a wall"""
        alternatives = []
        
        # Check all directions except the blocked one
        for direction in range(4):
            if direction != current_dir and not self.is_facing_wall(pos, direction):
                alternatives.append(direction)
        
        return alternatives

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def evaluate_position(self, pos: Tuple[int, int]) -> float:
        """Evaluate the reward value of a position"""
        if not self.is_valid_position(pos):
            return -1000
            
        base_reward = self.value[pos[0]][pos[1]]
        
        # Add penalties for visited positions
        visit_penalty = self.num_visited[pos[0]][pos[1]] * self.rewards['repeat']
        
        # Add guard penalty if applicable
        guard_penalty = self.guards_tracker.get(pos, 0)
        
        return base_reward + visit_penalty + guard_penalty

    def find_high_value_positions(self, center: Tuple[int, int], radius: int) -> List[Tuple[int, int]]:
        """Find positions with high rewards within radius"""
        candidates = []
        
        # First pass: look for high-value positions
        for i in range(max(0, center[0] - radius), min(self.size, center[0] + radius + 1)):
            for j in range(max(0, center[1] - radius), min(self.size, center[1] + radius + 1)):
                pos = (i, j)
                if pos == center:  # Skip current position
                    continue
                if self.manhattan_distance(center, pos) <= radius:
                    reward = self.evaluate_position(pos)
                    # Prioritize mission objectives and unexplored areas
                    if (reward > 0.5 or 
                        self.value[pos[0]][pos[1]] == self.rewards['hidden'] or
                        self.value[pos[0]][pos[1]] == self.rewards['mission']):
                        candidates.append((pos, reward))
        
        # Second pass: if no good candidates, look for any accessible position
        if not candidates:
            # Expand search radius and look for any reachable position
            expanded_radius = min(radius * 2, self.size // 2)
            for i in range(max(0, center[0] - expanded_radius), min(self.size, center[0] + expanded_radius + 1)):
                for j in range(max(0, center[1] - expanded_radius), min(self.size, center[1] + expanded_radius + 1)):
                    pos = (i, j)
                    if pos == center:
                        continue
                    # Check if position is reachable
                    if self.can_reach_position(center, pos):
                        reward = self.evaluate_position(pos)
                        # Include any position that's not terrible
                        if reward > -50:
                            candidates.append((pos, reward))
        
        # Third pass: if still no candidates, find positions to explore unknown areas
        if not candidates:
            candidates = self.find_exploration_targets(center)
        
        # Sort by reward and return top positions
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [pos for pos, _ in candidates[:min(15, len(candidates))]]

    def mcts_search(self, start_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Run MCTS to find the best goal position"""
        root = MCTSNode(start_pos)
        
        # Initialize untried actions (potential goal positions)
        high_value_positions = self.find_high_value_positions(start_pos, self.goal_search_radius)
        if not high_value_positions:
            return None
            
        root.untried_actions = high_value_positions.copy()
        
        for _ in range(self.mcts_iterations):
            node = self.mcts_select(root)
            if node is None:
                break
            reward = self.mcts_simulate(node)
            self.mcts_backpropagate(node, reward)
        
        if not root.children:
            # Fallback: return the best high-value position
            return high_value_positions[0] if high_value_positions else None
            
        # Select best child based on visit count and average reward
        best_child = max(root.children, key=lambda c: c.total_reward / max(c.visits, 1))
        return best_child.location

    def mcts_select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase of MCTS using UCT"""
        while not node.is_terminal:
            if not node.is_fully_expanded():
                return self.mcts_expand(node)
            else:
                best = node.best_child()
                if best is None:
                    node.is_terminal = True
                    break
                node = best
        return node

    def mcts_expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase of MCTS"""
        if not node.untried_actions:
            node.is_terminal = True
            return node
            
        action = random.choice(node.untried_actions)
        child = node.add_child(action, action)
        return child

    def mcts_simulate(self, node: MCTSNode) -> float:
        """Simulation phase - evaluate the position"""
        pos = node.location
        
        # Base reward for this position
        reward = self.evaluate_position(pos)
        
        # Add distance penalty (encourage closer goals)
        if node.parent:
            distance_penalty = -0.1 * self.manhattan_distance(node.parent.location, pos)
            reward += distance_penalty
        
        # Bonus for mission objectives (5-point cells)
        if self.value[pos[0]][pos[1]] == self.rewards['mission']:
            reward += 10  # Extra bonus for mission objectives
            
        return reward

    def mcts_backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagation phase of MCTS"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def astar_search(self, start: Tuple[int, int], goal: Tuple[int, int], avoid_guards: bool = False) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm"""
        if start == goal:
            return [start]
            
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in self.get_neighbors(current):
                # Skip if guard is present and we're avoiding guards
                if avoid_guards and neighbor in self.curr_guard_pos:
                    continue
                    
                tentative_g = g_score[current] + 1
                
                # Add extra cost for positions near guards when avoiding
                if avoid_guards:
                    for guard_pos in self.curr_guard_pos:
                        if self.manhattan_distance(neighbor, guard_pos) <= 2:
                            tentative_g += 5  # High penalty for being near guards
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found

    def find_escape_position(self, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Find position to escape to when guards are present"""
        best_pos = current_pos
        max_distance = 0
        
        # Find position that maximizes distance from all guards
        for i in range(self.size):
            for j in range(self.size):
                pos = (i, j)
                if not self.is_valid_position(pos):
                    continue
                    
                min_guard_distance = float('inf')
                for guard_pos in self.curr_guard_pos:
                    distance = self.manhattan_distance(pos, guard_pos)
                    min_guard_distance = min(min_guard_distance, distance)
                
                if min_guard_distance > max_distance:
                    # Check if we can actually reach this position
                    path = self.astar_search(current_pos, pos, avoid_guards=True)
                    if path:
                        max_distance = min_guard_distance
                        best_pos = pos
        
        return best_pos

    def get_next_action(self, current_pos: Tuple[int, int], current_dir: int) -> int:
        """Convert next position in path to action with wall handling"""
        # Check if we're facing a wall and handle it
        if self.is_facing_wall(current_pos, current_dir):
            # Find alternative directions
            alternatives = self.get_alternative_directions(current_pos, current_dir)
            if alternatives:
                # Choose the best alternative direction
                target_dir = alternatives[0]  # For now, just pick the first available
                
                # Calculate turn action needed
                if target_dir == (current_dir + 3) % 4:
                    return 2  # Turn left
                elif target_dir == (current_dir + 1) % 4:
                    return 3  # Turn right
                elif target_dir == (current_dir + 2) % 4:
                    return 2  # Turn left (will need another turn next step)
            
            # If no alternatives, turn around
            return 2  # Turn left to start turning around
        
        if not self.path_to_goal or len(self.path_to_goal) < 2:
            # No path available, try to move forward if possible
            if not self.is_facing_wall(current_pos, current_dir):
                return 0  # Move forward
            else:
                # Facing wall, turn
                return 2  # Turn left
            
        next_pos = self.path_to_goal[1]  # Next position in path
        
        # Calculate required direction to reach next position
        dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]
        
        target_dir = None
        for dir_idx, (ddx, ddy) in enumerate(self.DELTA):
            if dx == ddx and dy == ddy:
                target_dir = dir_idx
                break
        
        if target_dir is None:
            return 0  # Should not happen
        
        # Check if the intended direction is blocked by a wall
        if self.is_facing_wall(current_pos, target_dir):
            # Path is blocked, need to replan
            self.path_to_goal = []
            self.current_goal = None
            return 2  # Turn to start looking for new path
            
        # Calculate action needed
        if target_dir == current_dir:
            return 0  # Move forward
        elif target_dir == (current_dir + 2) % 4:
            return 1  # Move backward
        elif target_dir == (current_dir + 3) % 4:
            return 2  # Turn left
        elif target_dir == (current_dir + 1) % 4:
            return 3  # Turn right
        else:
            return 0  # Default

    def rl(self, observation_json: dict) -> int:
        viewcone = observation_json["viewcone"]
        direction = np.array(observation_json["direction"], np.uint8)
        location = np.array(observation_json["location"], np.uint8)
        self.step = observation_json["step"]

        if self.step == 0: 
            self.is_scout = bool(observation_json["scout"])
            self.load_rewards()

        self.update_map(viewcone, location, direction)
        
        current_pos = (int(location[0]), int(location[1]))
        current_dir = int(direction)
        
        # If guards are present, use A* to escape (no MCTS)
        if self.curr_guard_pos:
            escape_pos = self.find_escape_position(current_pos)
            self.path_to_goal = self.astar_search(current_pos, escape_pos, avoid_guards=True)
            self.current_goal = escape_pos
        else:
            # Check if we need a new goal or if current goal is reached
            if (self.current_goal is None or 
                current_pos == self.current_goal or 
                not self.path_to_goal or 
                current_pos not in self.path_to_goal or
                self.is_facing_wall(current_pos, current_dir)):  # Add wall check
                
                # Use MCTS to find new goal
                self.current_goal = self.mcts_search(current_pos)
                
                if self.current_goal:
                    # Use A* to plan path to goal
                    self.path_to_goal = self.astar_search(current_pos, self.current_goal)
                    
                    # If A* fails to find path, try alternative goals
                    if not self.path_to_goal:
                        # Try other high-value positions
                        alternatives = self.find_high_value_positions(current_pos, self.goal_search_radius)
                        for alt_goal in alternatives:
                            if alt_goal != self.current_goal:
                                test_path = self.astar_search(current_pos, alt_goal)
                                if test_path:
                                    self.current_goal = alt_goal
                                    self.path_to_goal = test_path
                                    break
                else:
                    # Fallback: find any reachable exploration target
                    exploration_targets = self.find_exploration_targets(current_pos)
                    if exploration_targets:
                        self.current_goal = exploration_targets[0]
                        self.path_to_goal = self.astar_search(current_pos, self.current_goal)
        
        # Update path if we've moved
        if self.path_to_goal and current_pos in self.path_to_goal:
            idx = self.path_to_goal.index(current_pos)
            self.path_to_goal = self.path_to_goal[idx:]
        
        # Get next action based on path
        return self.get_next_action(current_pos, current_dir)
import numpy as np
import math
import random
from enum import IntEnum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import heapq


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


@dataclass
class MCTSNode:
    """MCTS Node for strategic path planning"""
    goal_position: Tuple[int, int]
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    total_reward: float = 0.0
    untried_goals: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_goals is None:
            self.untried_goals = []
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_goals) == 0
    
    def is_terminal(self) -> bool:
        return len(self.untried_goals) == 0 and len(self.children) == 0
    
    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def add_child(self, goal_pos: Tuple[int, int]) -> 'MCTSNode':
        child = MCTSNode(goal_position=goal_pos, parent=self)
        self.children.append(child)
        if goal_pos in self.untried_goals:
            self.untried_goals.remove(goal_pos)
        return child
    
    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        return max(self.children, key=lambda child: child.ucb1_score(exploration_constant))


class AStarNavigator:
    """Enhanced A* pathfinding with guard avoidance and tactical navigation"""
    
    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size
        self.DELTA = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.guard_danger_radius = 3
        self.guard_avoidance_multiplier = 5.0
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
    
    def calculate_guard_danger(self, pos: Tuple[int, int], guards: List[Tuple[int, int]], 
                          guards_tracker: Dict[Tuple[int, int], float]) -> float:
        """Calculate danger level from guards at given position"""
        if not guards and not guards_tracker:
            return 0.0
        
        danger = 0.0
        
        # Current visible guards (highest priority)
        for guard_pos in guards:
            dist = self.manhattan_distance(pos, guard_pos)
            if dist <= self.guard_danger_radius:
                # Exponential danger increase as we get closer
                danger += self.guard_avoidance_multiplier * (2.0 ** (self.guard_danger_radius - dist))
        
        # Historical guard positions (lower priority but still avoid)
        for guard_pos, strength in guards_tracker.items():
            if guard_pos not in guards:  # Don't double count current guards
                dist = self.manhattan_distance(pos, guard_pos)
                if dist <= self.guard_danger_radius:
                    # Scale by historical strength and distance
                    # Fix: Ensure exponent is non-negative, use float base
                    exponent = max(0, self.guard_danger_radius - dist - 1)
                    historical_danger = abs(strength) * 0.3 * (2.0 ** exponent)
                    danger += historical_danger
        
        return danger
    
    def get_neighbors(self, pos: Tuple[int, int], direction: int, walls: np.ndarray) -> List[Tuple[Tuple[int, int], int, int]]:
        """Returns list of (position, direction, base_cost) tuples"""
        x, y = pos
        neighbors = []
        
        # Forward movement
        if not walls[direction][x][y]:
            new_pos = (x + self.DELTA[direction][0], y + self.DELTA[direction][1])
            if self.is_valid_position(new_pos):
                neighbors.append((new_pos, direction, 1))
        
        # Backward movement (higher base cost)
        opp_dir = (direction + 2) % 4
        if not walls[opp_dir][x][y]:
            new_pos = (x + self.DELTA[opp_dir][0], y + self.DELTA[opp_dir][1])
            if self.is_valid_position(new_pos):
                neighbors.append((new_pos, direction, 3))  # Higher cost for backward
        
        # Turning (no position change, moderate cost)
        neighbors.append((pos, (direction + 1) % 4, 2))  # Turn right
        neighbors.append((pos, (direction + 3) % 4, 2))  # Turn left
        
        return neighbors
    
    def astar_path_with_avoidance(self, start_pos: Tuple[int, int], start_dir: int, 
                                  goal_pos: Tuple[int, int], walls: np.ndarray,
                                  guards: List[Tuple[int, int]] = None,
                                  guards_tracker: Dict[Tuple[int, int], float] = None,
                                  is_scout: bool = True) -> List[int]:
        """A* pathfinding with guard avoidance"""
        if start_pos == goal_pos:
            return []
        
        if guards is None:
            guards = []
        if guards_tracker is None:
            guards_tracker = {}
        
        # Priority queue: (f_score, g_score, position, direction, path)
        heap = [(self.manhattan_distance(start_pos, goal_pos), 0, start_pos, start_dir, [])]
        visited = set()
        
        while heap:
            f_score, g_score, current_pos, current_dir, path = heapq.heappop(heap)
            
            state_key = (current_pos, current_dir)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            if current_pos == goal_pos:
                return path
            
            for next_pos, next_dir, base_cost in self.get_neighbors(current_pos, current_dir, walls):
                next_state_key = (next_pos, next_dir)
                if next_state_key in visited:
                    continue
                
                # Calculate movement cost with guard avoidance
                move_cost = base_cost
                
                if is_scout:
                    # Add guard danger cost for scouts
                    guard_danger = self.calculate_guard_danger(next_pos, guards, guards_tracker)
                    move_cost += guard_danger
                    
                    # Extra penalty for getting too close to guards
                    for guard_pos in guards:
                        if self.manhattan_distance(next_pos, guard_pos) <= 1:
                            move_cost += 50  # Very high penalty for adjacent to guard
                
                new_g_score = g_score + move_cost
                h_score = self.manhattan_distance(next_pos, goal_pos)
                new_f_score = new_g_score + h_score
                
                # Determine action taken
                if next_pos != current_pos:
                    # Movement action
                    if next_dir == current_dir:
                        action = 0  # Forward
                    else:
                        action = 1  # Backward
                else:
                    # Turning action
                    if next_dir == (current_dir + 1) % 4:
                        action = 3  # Turn right
                    else:
                        action = 2  # Turn left
                
                new_path = path + [action]
                heapq.heappush(heap, (new_f_score, new_g_score, next_pos, next_dir, new_path))
        
        return []  # No path found
    
    def find_escape_position(self, current_pos: Tuple[int, int], guards: List[Tuple[int, int]], 
                            walls: np.ndarray, max_distance: int = 8) -> Tuple[int, int]:
        """Find a safe position away from guards"""
        best_pos = current_pos
        best_distance = 0
        
        # Search for positions that maximize distance from guards
        for dx in range(-max_distance, max_distance + 1):
            for dy in range(-max_distance, max_distance + 1):
                if dx == 0 and dy == 0:
                    continue
                
                candidate_pos = (current_pos[0] + dx, current_pos[1] + dy)
                if not self.is_valid_position(candidate_pos):
                    continue
                
                # Calculate minimum distance to any guard
                if guards:
                    min_guard_dist = min(self.manhattan_distance(candidate_pos, guard_pos) 
                                       for guard_pos in guards)
                    
                    # Prefer positions further from guards
                    if min_guard_dist > best_distance:
                        best_distance = min_guard_dist
                        best_pos = candidate_pos
        
        return best_pos


class MCTSGoalPlanner:
    """MCTS-based strategic path planner"""
    
    def __init__(self, grid_size: int = 16, exploration_constant: float = 2):
        self.grid_size = grid_size
        self.exploration_constant = exploration_constant
        self.max_simulation_depth = 8
        self.goal_search_radius = 10
    
    def get_candidate_goals(self, current_pos: Tuple[int, int], 
                          value_map: np.ndarray, guards: List[Tuple[int, int]],
                          is_scout: bool = True) -> List[Tuple[int, int]]:
        """Generate candidate goal positions based on value and safety"""
        candidates = []
        x, y = current_pos
        
        # Search in expanding radius
        for radius in range(2, min(self.goal_search_radius, self.grid_size)):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius:  # Manhattan distance constraint
                        continue
                    
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < self.grid_size and 
                        0 <= new_y < self.grid_size and 
                        (new_x, new_y) != current_pos):
                        
                        value = value_map[new_x][new_y]
                        
                        # For scouts, avoid positions too close to guards
                        if is_scout and guards:
                            min_guard_dist = min(abs(new_x - gx) + abs(new_y - gy) 
                                               for gx, gy in guards)
                            if min_guard_dist < 3:  # Too close to guards
                                continue
                        
                        # Consider high-value positions and unexplored areas
                        if value > 0.3 or value == -1:  # Adjusted threshold
                            candidates.append((new_x, new_y))
        
        # Sort by value and safety
        def score_candidate(pos):
            value_score = value_map[pos[0]][pos[1]]
            if is_scout and guards:
                # Add safety bonus for scouts
                min_guard_dist = min(abs(pos[0] - gx) + abs(pos[1] - gy) 
                                   for gx, gy in guards)
                safety_bonus = min(min_guard_dist * 0.2, 1.0)
                return value_score + safety_bonus
            return value_score
        
        candidates.sort(key=score_candidate, reverse=True)
        return candidates[:15]  # Limit candidates
    
    def selection(self, node: MCTSNode) -> MCTSNode:
        """UCT selection phase"""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expansion(node)
            else:
                node = node.best_child(self.exploration_constant)
        return node
    
    def expansion(self, node: MCTSNode) -> MCTSNode:
        """Expand node with new child"""
        if node.untried_goals:
            goal_pos = random.choice(node.untried_goals)
            return node.add_child(goal_pos)
        return node
    
    def simulation(self, node: MCTSNode, current_pos: Tuple[int, int], 
                   value_map: np.ndarray, guards: List[Tuple[int, int]], 
                   is_scout: bool = True) -> float:
        """Monte Carlo simulation from node"""
        total_reward = 0.0
        
        # Reward for reaching the goal
        goal_reward = value_map[node.goal_position[0]][node.goal_position[1]]
        manhattan_dist = abs(current_pos[0] - node.goal_position[0]) + abs(current_pos[1] - node.goal_position[1])
        
        # Distance and safety factors
        distance_factor = max(0, 1.0 - manhattan_dist / (self.grid_size * 1.5))
        total_reward += goal_reward * distance_factor
        
        # Safety bonus for scouts
        if is_scout and guards:
            min_guard_dist = min(abs(node.goal_position[0] - gx) + abs(node.goal_position[1] - gy) 
                               for gx, gy in guards)
            safety_reward = min(min_guard_dist * 0.3, 2.0)
            total_reward += safety_reward
        
        # Simulate random exploration from goal
        sim_pos = node.goal_position
        visited_positions = {sim_pos}
        
        for step in range(self.max_simulation_depth):
            valid_moves = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = sim_pos[0] + dx, sim_pos[1] + dy
                if (0 <= new_x < self.grid_size and 
                    0 <= new_y < self.grid_size):
                    valid_moves.append((new_x, new_y))
            
            if valid_moves:
                # Prefer unvisited positions
                unvisited = [pos for pos in valid_moves if pos not in visited_positions]
                if unvisited:
                    sim_pos = random.choice(unvisited)
                else:
                    sim_pos = random.choice(valid_moves)
                
                visited_positions.add(sim_pos)
                
                # Add discounted position value
                pos_value = value_map[sim_pos[0]][sim_pos[1]]
                discount = 0.7 ** step
                total_reward += pos_value * discount
                
                # Safety consideration in simulation
                if is_scout and guards:
                    min_guard_dist = min(abs(sim_pos[0] - gx) + abs(sim_pos[1] - gy) 
                                       for gx, gy in guards)
                    if min_guard_dist < 2:
                        total_reward -= 2.0 * discount  # Penalty for dangerous positions
        
        return total_reward
    
    def backpropagation(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def mcts_search(self, current_pos: Tuple[int, int], value_map: np.ndarray,
                    guards: List[Tuple[int, int]], is_scout: bool = True,
                    iterations: int = 80) -> Tuple[int, int]:
        """Main MCTS search for strategic goal selection"""
        
        # Get candidate goals
        candidate_goals = self.get_candidate_goals(current_pos, value_map, guards, is_scout)
        
        if not candidate_goals:
            # No good goals found, return a safe nearby position
            if is_scout and guards:
                # Move away from guards
                best_pos = current_pos
                best_distance = 0
                for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    if (0 <= new_pos[0] < self.grid_size and 
                        0 <= new_pos[1] < self.grid_size):
                        min_guard_dist = min(abs(new_pos[0] - gx) + abs(new_pos[1] - gy) 
                                           for gx, gy in guards)
                        if min_guard_dist > best_distance:
                            best_distance = min_guard_dist
                            best_pos = new_pos
                return best_pos
            else:
                # Default nearby exploration
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    if (0 <= new_pos[0] < self.grid_size and 
                        0 <= new_pos[1] < self.grid_size):
                        return new_pos
                return current_pos
        
        # Initialize root node
        root = MCTSNode(goal_position=current_pos, untried_goals=candidate_goals.copy())
        
        # MCTS iterations
        for _ in range(iterations):
            leaf_node = self.selection(root)
            reward = self.simulation(leaf_node, current_pos, value_map, guards, is_scout)
            self.backpropagation(leaf_node, reward)
        
        # Select best goal
        if root.children:
            best_child = max(root.children, key=lambda child: child.total_reward / max(child.visits, 1))
            return best_child.goal_position
        else:
            return candidate_goals[0] if candidate_goals else current_pos

class RLManager:
    """Enhanced RL Manager with A*-only navigation and MCTS strategic planning"""
    
    SCOUT_REWARDS = {
        'empty': 0,
        'hidden': 1.2,
        'recon': 1,
        'mission': 10,
        'guard': -100,  # Increased penalty
        'repeat': -2,
        'wall': -0.1,
        'backwards': -0.3
    }

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

    DELTA = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self):
        self.size = 16
        self.rewards = None
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
        self.decay_factor = 0.9
        self.scout_pos: Optional[Tuple[int, int]] = None
        self.decay_multiplier = 1
        
        # Navigation components
        self.mcts_planner = MCTSGoalPlanner(grid_size=self.size)
        self.astar_navigator = AStarNavigator(grid_size=self.size)
        
        # Goal planning state
        self.current_goal: Optional[Tuple[int, int]] = None
        self.goal_path: List[int] = []
        self.goal_replanning_interval = 8  # Replan more frequently
        self.last_goal_planning_step = -1
        self.emergency_mode = False  # Emergency escape mode
        
        # ADD: Anti-oscillation tracking
        self.action_history = []  # Track last few actions
        self.position_history = []  # Track last few positions  
        self.stuck_counter = 0  # Count how many steps we've been stuck
        self.last_position = None
        self.last_direction = None
        
        # ADD: Fallback exploration for when stuck
        self.fallback_mode = False
        self.fallback_steps = 0
        self.max_fallback_steps = 15
        
        # ADD: Direction preference to avoid wall-facing
        self.direction_stuck_counter = 0
        self.last_successful_direction = None

    def load_rewards(self):
        if self.is_scout:
            self.rewards = self.SCOUT_REWARDS
        else:
            self.rewards = self.GUARD_REWARDS
        
        self.value = np.full((self.size, self.size), self.rewards['hidden'])

    def update_decay_factor(self, num_guards_spotted: int = 0) -> None:
        self.decay_factor = max(0.5, 1 - (1 + 3*num_guards_spotted)/15)

    def is_in_danger(self, current_pos: Tuple[int, int]) -> bool:
        """Check if current position is dangerous for scouts"""
        if not self.is_scout or not self.curr_guard_pos:
            return False
        
        for guard_pos in self.curr_guard_pos:
            if abs(current_pos[0] - guard_pos[0]) + abs(current_pos[1] - guard_pos[1]) <= 3:
                return True
        return False

    def update_map(self, viewcone: List[List[int]], location: Tuple[int, int], direction: int) -> None:
        """Update map based on viewcone observation"""
        new_grid = np.array(viewcone, dtype=np.uint8)
        curr_direction = np.array(direction, dtype=np.int8)
        curr_location = np.array(location, dtype=np.int8)

        self.num_visited[curr_location[0]][curr_location[1]] += 1

        # Rotate clockwise so absolute north faces up
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

    def detect_stuck_or_oscillating(self, current_pos: Tuple[int, int], current_dir: int) -> bool:
        """Detect if agent is stuck or oscillating"""
        # Update position history
        self.position_history.append(current_pos)
        if len(self.position_history) > 8:
            self.position_history.pop(0)
        
        # Check if we haven't moved in several steps
        if self.last_position == current_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        # Check if facing wall and can't move forward
        wall_ahead = self.obs_wall[current_dir][current_pos[0]][current_pos[1]]
        if wall_ahead:
            self.direction_stuck_counter += 1
        else:
            self.direction_stuck_counter = 0
        
        # Check for oscillation (returning to same positions repeatedly)
        if len(self.position_history) >= 6:
            recent_positions = self.position_history[-6:]
            unique_positions = set(recent_positions)
            if len(unique_positions) <= 2:  # Only visiting 1-2 positions
                return True
        
        # Check action oscillation (turning back and forth)
        if len(self.action_history) >= 6:
            recent_actions = self.action_history[-6:]
            # Check for repetitive turn patterns
            turn_actions = [a for a in recent_actions if a in [2, 3]]
            if len(turn_actions) >= 4:
                # Check if alternating between left and right turns
                if len(set(turn_actions[-4:])) <= 2:
                    return True
        
        # Stuck conditions
        return (self.stuck_counter >= 3 or 
                self.direction_stuck_counter >= 2 or 
                (wall_ahead and len(self.action_history) >= 3 and 
                 all(a in [2, 3] for a in self.action_history[-3:])))

    def get_unstuck_action(self, current_pos: Tuple[int, int], current_dir: int) -> int:
        """Get action to become unstuck"""
        # Priority 1: Try to move in any direction that's not blocked
        movement_options = []
        
        # Check forward movement
        if not self.obs_wall[current_dir][current_pos[0]][current_pos[1]]:
            forward_pos = (current_pos[0] + self.DELTA[current_dir][0], 
                          current_pos[1] + self.DELTA[current_dir][1])
            if (self.astar_navigator.is_valid_position(forward_pos) and 
                forward_pos not in self.position_history[-4:]):
                movement_options.append(0)  # Forward
        
        # Check backward movement
        backward_dir = (current_dir + 2) % 4
        if not self.obs_wall[backward_dir][current_pos[0]][current_pos[1]]:
            backward_pos = (current_pos[0] + self.DELTA[backward_dir][0], 
                           current_pos[1] + self.DELTA[backward_dir][1])
            if (self.astar_navigator.is_valid_position(backward_pos) and 
                backward_pos not in self.position_history[-3:]):
                movement_options.append(1)  # Backward
        
        # Priority 2: If we can move, prefer forward unless we've been going forward too much
        if movement_options:
            if 0 in movement_options:
                # Check if we've been going forward too much
                recent_forwards = sum(1 for a in self.action_history[-4:] if a == 0)
                if recent_forwards < 3:
                    return 0
            
            return random.choice(movement_options)
        
        # Priority 3: Turn to find a clear direction
        # Try turning to directions we haven't tried recently
        turn_preferences = []
        
        # Check right turn
        right_dir = (current_dir + 1) % 4
        if not self.obs_wall[right_dir][current_pos[0]][current_pos[1]]:
            turn_preferences.append(3)  # Turn right
        
        # Check left turn  
        left_dir = (current_dir - 1) % 4
        if not self.obs_wall[left_dir][current_pos[0]][current_pos[1]]:
            turn_preferences.append(2)  # Turn left
        
        # Prefer the direction that hasn't been tried recently
        if turn_preferences:
            recent_turns = self.action_history[-4:]
            if 3 not in recent_turns and 3 in turn_preferences:
                return 3
            elif 2 not in recent_turns and 2 in turn_preferences:
                return 2
            else:
                return random.choice(turn_preferences)
        
        # Priority 4: Force turn even if no clear path (to break wall-facing)
        # Alternate between left and right to systematically explore
        if len(self.action_history) % 2 == 0:
            return 3  # Turn right
        else:
            return 2  # Turn left

    def should_replan_goal(self, current_pos: Tuple[int, int], current_dir: int) -> bool:
        """Determine if we should replan our goal"""
        # Always replan if no current goal
        if self.current_goal is None:
            return True
        
        # Replan if we've reached our goal
        if current_pos == self.current_goal:
            return True
        
        # Replan if stuck or oscillating
        if self.detect_stuck_or_oscillating(current_pos, current_dir):
            return True
        
        # Replan if path is empty and we haven't reached goal
        if not self.goal_path and current_pos != self.current_goal:
            return True
        
        # Replan if we're in emergency mode (danger detected)
        if self.emergency_mode:
            return True
        
        # Replan periodically for better goals
        steps_since_last_plan = self.step - self.last_goal_planning_step
        if steps_since_last_plan >= self.goal_replanning_interval:
            return True
        
        # Replan if goal is no longer valuable
        if (self.current_goal and 
            self.value[self.current_goal[0]][self.current_goal[1]] <= 0):
            return True
        
        # Replan if we've been trying to reach the same goal for too long
        if steps_since_last_plan >= 20:
            return True
        
        return False

    def get_action(self, current_pos: Tuple[int, int], current_dir: int) -> int:
        """Main action selection logic"""
        self.step += 1
        
        # Update history
        if self.last_position is not None:
            self.action_history.append(self.last_position)
        if len(self.action_history) > 10:
            self.action_history.pop(0)
        
        # Check for emergency situations (scout near guards)
        if self.is_scout and self.is_in_danger(current_pos):
            self.emergency_mode = True
        else:
            self.emergency_mode = False
        
        # Handle fallback mode
        if self.fallback_mode:
            self.fallback_steps += 1
            if self.fallback_steps >= self.max_fallback_steps:
                self.fallback_mode = False
                self.fallback_steps = 0
            else:
                # Simple random exploration
                valid_actions = []
                for action in [0, 1, 2, 3]:
                    if action == 0:  # forward
                        if not self.obs_wall[current_dir][current_pos[0]][current_pos[1]]:
                            valid_actions.append(action)
                    elif action == 1:  # backward
                        backward_dir = (current_dir + 2) % 4
                        if not self.obs_wall[backward_dir][current_pos[0]][current_pos[1]]:
                            valid_actions.append(action)
                    else:  # turns
                        valid_actions.append(action)
                
                if valid_actions:
                    chosen_action = random.choice(valid_actions)
                    self.last_position = current_pos
                    self.last_direction = current_dir
                    return chosen_action
        
        # Check if we need to get unstuck
        if self.detect_stuck_or_oscillating(current_pos, current_dir):
            # Enter fallback mode if we've been stuck too long
            if self.stuck_counter >= 5:
                self.fallback_mode = True
                self.fallback_steps = 0
            
            action = self.get_unstuck_action(current_pos, current_dir)
            self.last_position = current_pos
            self.last_direction = current_dir
            return action
        
        # Strategic planning with MCTS
        if self.should_replan_goal(current_pos, current_dir):
            self.current_goal = self.mcts_planner.mcts_search(
                current_pos, self.value, self.curr_guard_pos, 
                is_scout=self.is_scout, iterations=60
            )
            self.last_goal_planning_step = self.step
            
            # Plan path to new goal
            if self.current_goal and self.current_goal != current_pos:
                self.goal_path = self.astar_navigator.astar_path_with_avoidance(
                    current_pos, current_dir, self.current_goal, self.obs_wall,
                    self.curr_guard_pos, self.guards_tracker, is_scout=self.is_scout
                )
        
        # Execute planned path
        if self.goal_path:
            action = self.goal_path.pop(0)
            self.last_position = current_pos
            self.last_direction = current_dir
            return action
        
        # Fallback: simple exploration
        exploration_actions = []
        
        # Prefer forward movement if not blocked
        if not self.obs_wall[current_dir][current_pos[0]][current_pos[1]]:
            forward_pos = (current_pos[0] + self.DELTA[current_dir][0], 
                          current_pos[1] + self.DELTA[current_dir][1])
            if (self.astar_navigator.is_valid_position(forward_pos) and 
                self.num_visited[forward_pos[0]][forward_pos[1]] < 3):
                exploration_actions.extend([0] * 3)  # Weight forward movement
        
        # Add turn options
        exploration_actions.extend([2, 3])
        
        # Add backward if really needed
        backward_dir = (current_dir + 2) % 4
        if not self.obs_wall[backward_dir][current_pos[0]][current_pos[1]]:
            exploration_actions.append(1)
        
        if exploration_actions:
            action = random.choice(exploration_actions)
        else:
            # Last resort: just turn
            action = random.choice([2, 3])
        
        self.last_position = current_pos
        self.last_direction = current_dir
        return action

    def rl(self, observation_json: dict) -> int:
        viewcone = observation_json["viewcone"]
        direction = np.array(observation_json["direction"], np.uint8)
        location = np.array(observation_json["location"], np.int32)  # Changed to int32 to handle negative arithmetic
        self.step = observation_json["step"]
        
        if self.step == 0: 
            self.is_scout = bool(observation_json["scout"])
            self.load_rewards()
        
        self.update_map(viewcone, (int(location[0]), int(location[1])), int(direction))
        
        action = self.get_action((int(location[0]), int(location[1])), int(direction))
        
        return action
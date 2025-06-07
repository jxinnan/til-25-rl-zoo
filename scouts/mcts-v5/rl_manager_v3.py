import numpy as np
import heapq
from enum import IntEnum
from typing import List, Tuple, Optional, Dict
import random


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


class GuardMemory:
    """Tracks dangerous locations where guards have been encountered"""
    
    def __init__(self, memory_duration: int = 50):
        self.dangerous_locations: Dict[Tuple[int, int], int] = {}  # location -> step_last_seen
        self.memory_duration = memory_duration
    
    def add_dangerous_location(self, location: Tuple[int, int], current_step: int):
        """Mark a location as dangerous"""
        self.dangerous_locations[location] = current_step
    
    def is_location_dangerous(self, location: Tuple[int, int], current_step: int) -> bool:
        """Check if a location is considered dangerous"""
        if location not in self.dangerous_locations:
            return False
        
        steps_since_seen = current_step - self.dangerous_locations[location]
        return steps_since_seen < self.memory_duration
    
    def get_danger_penalty(self, location: Tuple[int, int], current_step: int) -> float:
        """Get penalty for being near a dangerous location"""
        if not self.is_location_dangerous(location, current_step):
            return 0
        
        steps_since_seen = current_step - self.dangerous_locations[location]
        # Exponential decay of danger over time
        decay_factor = (self.memory_duration - steps_since_seen) / self.memory_duration
        return 30 * decay_factor  # Base penalty of 30, decaying over time
    
    def cleanup_old_memories(self, current_step: int):
        """Remove old dangerous location memories"""
        expired_locations = [
            loc for loc, step in self.dangerous_locations.items()
            if current_step - step >= self.memory_duration
        ]
        for loc in expired_locations:
            del self.dangerous_locations[loc]


class AStar:
    """A* pathfinding with rotation costs and guard memory"""
    
    def __init__(self, walls: np.ndarray, size: int = 16, guard_positions: List[Tuple[int, int]] = None, 
                 avoidance_radius: int = 2, guard_memory: GuardMemory = None, current_step: int = 0):
        self.walls = walls
        self.size = size
        self.DELTA = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.guard_positions = guard_positions or []
        self.avoidance_radius = avoidance_radius
        self.guard_memory = guard_memory
        self.current_step = current_step
    
    def is_near_guard(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within avoidance radius of any guard"""
        for guard_pos in self.guard_positions:
            distance = abs(pos[0] - guard_pos[0]) + abs(pos[1] - guard_pos[1])
            if distance <= self.avoidance_radius:
                return True
        return False
    
    def get_location_penalty(self, pos: Tuple[int, int]) -> float:
        """Get penalty for a location based on guard memory and current guards"""
        penalty = 0
        
        # Current guard penalty
        if self.is_near_guard(pos):
            penalty += 100
        
        # Historical guard penalty
        if self.guard_memory:
            penalty += self.guard_memory.get_danger_penalty(pos, self.current_step)
        
        return penalty
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Manhattan distance heuristic with guard penalty"""
        base_distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        guard_penalty = self.get_location_penalty(pos)
        return base_distance + guard_penalty
    
    def get_neighbors(self, state: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], int]]:
        """Get valid neighbors with their costs"""
        x, y, direction = state
        neighbors = []
        
        # Forward movement (cost 1)
        if not self.walls[direction][x][y]:
            new_x = int(x + self.DELTA[direction][0])
            new_y = int(y + self.DELTA[direction][1])
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                move_cost = 1 + self.get_location_penalty((new_x, new_y))
                neighbors.append(((new_x, new_y, direction), move_cost))
        
        # Backward movement (cost 1)
        back_dir = (direction + 2) % 4
        if not self.walls[back_dir][x][y]:
            new_x = int(x + self.DELTA[back_dir][0])
            new_y = int(y + self.DELTA[back_dir][1])
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                move_cost = 1 + self.get_location_penalty((new_x, new_y))
                neighbors.append(((new_x, new_y, direction), move_cost))
        
        # Turn left (cost 2)
        left_dir = (direction + 3) % 4
        neighbors.append(((x, y, left_dir), 2))
        
        # Turn right (cost 2)
        right_dir = (direction + 1) % 4
        neighbors.append(((x, y, right_dir), 2))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int, int], goal: Tuple[int, int]) -> Optional[List[int]]:
        """
        Find path from start (x, y, direction) to goal (x, y)
        Returns list of actions: 0=forward, 1=backward, 2=turn_left, 3=turn_right
        """
        start_x, start_y, start_dir = start
        goal_x, goal_y = goal
        
        # Priority queue: (f_cost, g_cost, state)
        open_set = [(self.heuristic((start_x, start_y), goal), 0, start)]
        came_from = {}
        g_cost = {start: 0}
        
        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if current[0] == goal_x and current[1] == goal_y:
                # Reconstruct path
                path = []
                while current in came_from:
                    prev_state, action = came_from[current]
                    path.append(action)
                    current = prev_state
                return list(reversed(path))
            
            # Explore neighbors
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g = g_cost[current] + move_cost
                
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    # Determine action taken
                    if neighbor[0] != current[0] or neighbor[1] != current[1]:
                        # Movement occurred
                        if neighbor[2] == current[2]:  # Same direction
                            # Check if forward or backward
                            expected_forward = (
                                int(current[0] + self.DELTA[current[2]][0]),
                                int(current[1] + self.DELTA[current[2]][1])
                            )
                            action = 0 if (neighbor[0], neighbor[1]) == expected_forward else 1
                        else:
                            # This shouldn't happen in our movement model
                            continue
                    else:
                        # Rotation occurred
                        if neighbor[2] == (current[2] + 3) % 4:
                            action = 2  # Turn left
                        else:
                            action = 3  # Turn right
                    
                    came_from[neighbor] = (current, action)
                    g_cost[neighbor] = tentative_g
                    f_cost = tentative_g + self.heuristic((neighbor[0], neighbor[1]), goal)
                    heapq.heappush(open_set, (f_cost, tentative_g, neighbor))
        
        return None  # No path found
    
    def find_safe_position(self, start: Tuple[int, int, int], min_distance: int = 3) -> Optional[Tuple[int, int]]:
        """Find a safe position away from all guards and dangerous locations"""
        start_x, start_y, start_dir = start
        safe_positions = []
        
        # Find all positions that are safe
        for x in range(self.size):
            for y in range(self.size):
                is_safe = True
                
                # Check distance from current guards
                for guard_pos in self.guard_positions:
                    distance = abs(x - guard_pos[0]) + abs(y - guard_pos[1])
                    if distance < min_distance:
                        is_safe = False
                        break
                
                # Check if location has dangerous history
                if is_safe and self.guard_memory:
                    if self.guard_memory.is_location_dangerous((x, y), self.current_step):
                        # Allow but penalize dangerous locations if they're far enough
                        for guard_pos in self.guard_positions:
                            distance = abs(x - guard_pos[0]) + abs(y - guard_pos[1])
                            if distance < min_distance + 2:  # Extra buffer for dangerous locations
                                is_safe = False
                                break
                
                if is_safe:
                    # Calculate distance from start position
                    start_distance = abs(x - start_x) + abs(y - start_y)
                    
                    # Add penalty for dangerous locations
                    danger_penalty = 0
                    if self.guard_memory:
                        danger_penalty = self.guard_memory.get_danger_penalty((x, y), self.current_step)
                    
                    safe_positions.append((start_distance + danger_penalty, (x, y)))
        
        if not safe_positions:
            return None
        
        # Sort by distance (including danger penalty) and return best position
        safe_positions.sort()
        return safe_positions[0][1]


class GuardAvoidanceManager:
    """Manages guard detection and avoidance behavior with memory"""
    
    def __init__(self, rl_manager):
        self.rl_manager = rl_manager
        self.guard_memory = GuardMemory(memory_duration=50)  # Remember for 50 steps
        self.last_known_guards = []
        self.avoidance_active = False
        self.avoidance_target = None
        self.avoidance_path = []
        self.avoidance_path_index = 0
        self.min_safe_distance = 3
        self.last_guard_encounter_step = -1
    
    def detect_guards(self) -> List[Tuple[int, int]]:
        """Detect currently visible guards"""
        return self.rl_manager.curr_guard_pos
    
    def guards_detected(self) -> bool:
        """Check if guards are currently visible"""
        return len(self.rl_manager.curr_guard_pos) > 0
    
    def update_guard_memory(self, current_step: int):
        """Update guard memory with current guard positions"""
        if self.guards_detected():
            for guard_pos in self.rl_manager.curr_guard_pos:
                self.guard_memory.add_dangerous_location(guard_pos, current_step)
                # Also mark nearby positions as dangerous
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nearby_pos = (guard_pos[0] + dx, guard_pos[1] + dy)
                        if (0 <= nearby_pos[0] < self.rl_manager.size and 
                            0 <= nearby_pos[1] < self.rl_manager.size):
                            self.guard_memory.add_dangerous_location(nearby_pos, current_step)
            
            self.last_guard_encounter_step = current_step
        
        # Cleanup old memories
        self.guard_memory.cleanup_old_memories(current_step)
    
    def should_activate_avoidance(self, current_pos: Tuple[int, int]) -> bool:
        """Determine if avoidance behavior should be activated"""
        if not self.guards_detected():
            return False
        
        # Check if any guard is too close
        for guard_pos in self.rl_manager.curr_guard_pos:
            distance = abs(current_pos[0] - guard_pos[0]) + abs(current_pos[1] - guard_pos[1])
            if distance <= self.min_safe_distance:
                return True
        
        return False
    
    def is_in_dangerous_area(self, current_pos: Tuple[int, int]) -> bool:
        """Check if current position is in a historically dangerous area"""
        return self.guard_memory.is_location_dangerous(current_pos, self.rl_manager.step)
    
    def plan_avoidance(self, current_pos: Tuple[int, int], current_dir: int) -> bool:
        """Plan an avoidance route away from detected guards and dangerous areas"""
        # Create A* with current guard positions and memory
        astar = AStar(
            self.rl_manager.obs_wall, 
            self.rl_manager.size, 
            self.rl_manager.curr_guard_pos,
            avoidance_radius=2,
            guard_memory=self.guard_memory,
            current_step=self.rl_manager.step
        )
        
        # Find a safe position
        safe_position = astar.find_safe_position(
            (current_pos[0], current_pos[1], current_dir),
            min_distance=self.min_safe_distance
        )
        
        if safe_position is None:
            return False
        
        # Plan path to safe position
        path = astar.find_path(
            (current_pos[0], current_pos[1], current_dir),
            safe_position
        )
        
        if path is None:
            return False
        
        self.avoidance_target = safe_position
        self.avoidance_path = path
        self.avoidance_path_index = 0
        self.avoidance_active = True
        
        return True
    
    def get_avoidance_action(self) -> Optional[int]:
        """Get next action for avoidance behavior"""
        if not self.avoidance_active or self.avoidance_path_index >= len(self.avoidance_path):
            return None
        
        action = self.avoidance_path[self.avoidance_path_index]
        self.avoidance_path_index += 1
        
        return action
    
    def is_avoidance_complete(self, current_pos: Tuple[int, int]) -> bool:
        """Check if avoidance is complete"""
        if not self.avoidance_active:
            return True
        
        # Check if we reached the avoidance target
        if self.avoidance_target and current_pos == self.avoidance_target:
            return True
        
        # Check if path is complete
        if self.avoidance_path_index >= len(self.avoidance_path):
            return True
        
        return False
    
    def clear_avoidance(self):
        """Clear current avoidance state"""
        self.avoidance_active = False
        self.avoidance_target = None
        self.avoidance_path = []
        self.avoidance_path_index = 0


class WaypointPlanner:
    """Plans optimal waypoints based on current map state with guard memory awareness"""
    
    def __init__(self, rl_manager):
        self.rl_manager = rl_manager
        self.DELTA = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    def calculate_waypoint_value(self, waypoint: Tuple[int, int], current_pos: Tuple[int, int]) -> float:
        """Calculate the value/attractiveness of a waypoint"""
        x, y = waypoint
        
        # Base value from the map
        base_value = self.rl_manager.value[x][y]
        
        # Distance penalty (prefer closer waypoints)
        distance = abs(x - current_pos[0]) + abs(y - current_pos[1])
        distance_penalty = distance * 0.5
        
        # Visit frequency penalty (discourage over-visiting)
        visit_penalty = self.rl_manager.num_visited[x][y] * self.rl_manager.rewards['repeat']
        
        # Current guard avoidance penalty
        guard_penalty = 0
        for guard_pos in self.rl_manager.curr_guard_pos:
            guard_distance = abs(x - guard_pos[0]) + abs(y - guard_pos[1])
            if guard_distance <= 3:
                guard_penalty += 50 * (4 - guard_distance)
        
        # Historical danger penalty
        danger_penalty = 0
        if hasattr(self.rl_manager.navigation_manager.guard_avoidance, 'guard_memory'):
            guard_memory = self.rl_manager.navigation_manager.guard_avoidance.guard_memory
            danger_penalty = guard_memory.get_danger_penalty(waypoint, self.rl_manager.step)
        
        # Role-specific bonuses
        if self.rl_manager.is_scout:
            guard_bonus = self.rl_manager.guards_tracker.get(waypoint, 0)
            return base_value + guard_bonus + visit_penalty - distance_penalty - guard_penalty - danger_penalty
        else:
            scout_bonus = 0
            if self.rl_manager.scout_pos == waypoint:
                scout_bonus = self.rl_manager.rewards['scout'] * self.rl_manager.decay_multiplier
            
            target_bonus = 0
            if self.rl_manager.step < self.rl_manager.rewards['preferential_turns']:
                target = self.rl_manager.rewards['target']
                manhattan_dist = abs(x - target[0]) + abs(y - target[1])
                target_bonus = self.rl_manager.rewards['preferential_drive'] * manhattan_dist
            
            return base_value + scout_bonus + target_bonus + visit_penalty - distance_penalty - guard_penalty - danger_penalty
    
    def get_visible_waypoints(self, current_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get list of visible/explored positions that could serve as waypoints"""
        waypoints = []
        
        for i in range(self.rl_manager.size):
            for j in range(self.rl_manager.size):
                if self.rl_manager.value[i][j] != self.rl_manager.rewards['hidden']:
                    if (i, j) != current_pos:
                        dist = abs(i - current_pos[0]) + abs(j - current_pos[1])
                        if dist >= 1:
                            waypoints.append((i, j))
        
        return waypoints
    
    def find_best_waypoint(self, current_pos: Tuple[int, int], current_dir: int) -> Optional[Tuple[int, int]]:
        """Find the best waypoint based on current map state"""
        visible_waypoints = self.get_visible_waypoints(current_pos)
        
        if not visible_waypoints:
            return None
        
        # Calculate value for each waypoint and filter reachable ones
        waypoint_scores = []
        
        # Use A* with guard memory for pathfinding
        guard_memory = None
        if hasattr(self.rl_manager.navigation_manager.guard_avoidance, 'guard_memory'):
            guard_memory = self.rl_manager.navigation_manager.guard_avoidance.guard_memory
        
        astar = AStar(
            self.rl_manager.obs_wall, 
            self.rl_manager.size,
            guard_positions=self.rl_manager.curr_guard_pos,
            guard_memory=guard_memory,
            current_step=self.rl_manager.step
        )
        
        for waypoint in visible_waypoints:
            # Check if waypoint is reachable
            path = astar.find_path((current_pos[0], current_pos[1], current_dir), waypoint)
            if path is not None:
                value = self.calculate_waypoint_value(waypoint, current_pos)
                # Factor in path length (shorter paths are better)
                adjusted_value = value - len(path) * 0.1
                waypoint_scores.append((adjusted_value, waypoint))
        
        if not waypoint_scores:
            return None
        
        # Return the waypoint with highest score
        waypoint_scores.sort(key=lambda x: x[0], reverse=True)
        return waypoint_scores[0][1]


class NavigationManager:
    """Manages waypoint-based navigation with guard avoidance and memory"""
    
    def __init__(self, rl_manager):
        self.rl_manager = rl_manager
        self.waypoint_planner = WaypointPlanner(rl_manager)
        self.guard_avoidance = GuardAvoidanceManager(rl_manager)
        self.astar = None
        
        # Current navigation state
        self.current_waypoint = None
        self.current_path = []
        self.path_index = 0
    
    def initialize(self):
        """Initialize navigation components after RL manager is set up"""
        self.astar = AStar(self.rl_manager.obs_wall, self.rl_manager.size)
    
    def needs_new_waypoint(self, current_pos: Tuple[int, int]) -> bool:
        """Check if we need to find a new waypoint"""
        return (
            self.current_waypoint is None or
            self.current_path is None or
            self.path_index >= len(self.current_path) or
            current_pos == self.current_waypoint
        )
    
    def clear_current_path(self):
        """Clear current navigation path"""
        self.current_waypoint = None
        self.current_path = []
        self.path_index = 0
    
    def plan_navigation(self, current_pos: Tuple[int, int], current_dir: int) -> Optional[int]:
        """Plan navigation with guard avoidance priority and memory"""
        
        # Update guard memory with current step
        self.guard_avoidance.update_guard_memory(self.rl_manager.step)
        
        # PRIORITY 1: Handle guard avoidance
        if self.guard_avoidance.should_activate_avoidance(current_pos):
            # Clear any existing path when guards are detected
            self.clear_current_path()
            self.guard_avoidance.clear_avoidance()
            
            # Plan new avoidance route
            if self.guard_avoidance.plan_avoidance(current_pos, current_dir):
                print(f"GUARD DETECTED! Activating avoidance from {current_pos}")
        
        # Execute avoidance if active
        if self.guard_avoidance.avoidance_active:
            action = self.guard_avoidance.get_avoidance_action()
            if action is not None:
                return action
            
            # Check if avoidance is complete
            if self.guard_avoidance.is_avoidance_complete(current_pos):
                print(f"Avoidance complete at {current_pos}")
                self.guard_avoidance.clear_avoidance()
                # Clear current path to force re-planning with updated guard memory
                self.clear_current_path()
        
        # PRIORITY 2: Check if we're in a dangerous area and should avoid it
        if (not self.guard_avoidance.avoidance_active and 
            self.guard_avoidance.is_in_dangerous_area(current_pos) and
            not self.guard_avoidance.guards_detected()):
            
            # We're in a dangerous area but no guards are currently visible
            # Plan a route away from this dangerous area
            self.clear_current_path()
            if self.guard_avoidance.plan_avoidance(current_pos, current_dir):
                print(f"In dangerous area at {current_pos}, planning escape route")
        
        # PRIORITY 3: Normal waypoint navigation
        if self.needs_new_waypoint(current_pos):
            # Find best waypoint (now considers guard memory)
            self.current_waypoint = self.waypoint_planner.find_best_waypoint(current_pos, current_dir)
            
            if self.current_waypoint is None:
                return None
            
            # Plan path to waypoint using A* with guard memory
            guard_memory = self.guard_avoidance.guard_memory
            self.astar = AStar(
                self.rl_manager.obs_wall, 
                self.rl_manager.size,
                guard_positions=self.rl_manager.curr_guard_pos,
                guard_memory=guard_memory,
                current_step=self.rl_manager.step
            )
            
            self.current_path = self.astar.find_path(
                (current_pos[0], current_pos[1], current_dir), 
                self.current_waypoint
            )
            
            if self.current_path is None:
                self.current_waypoint = None
                return None
            
            self.path_index = 0
        
        # Follow current path
        if self.path_index < len(self.current_path):
            action = self.current_path[self.path_index]
            self.path_index += 1
            return action
        
        return None


# Updated RLManager class with integrated guard avoidance and memory
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
        self.guards_tracker = {}
        self.decay_factor = self.update_decay_factor()
        self.scout_pos = None
        self.decay_multiplier = 1
        
        # Navigation system with guard avoidance and memory
        self.navigation_manager = NavigationManager(self)

    def load_rewards(self):
        if self.is_scout:
            self.rewards = self.SCOUT_REWARDS
        else:
            self.rewards = self.GUARD_REWARDS
        
        self.value = np.full((self.size, self.size), self.rewards['hidden'])
        
        # Initialize navigation after rewards are loaded
        self.navigation_manager.initialize()

    def update_decay_factor(self, num_guards_spotted: int = 0) -> None:
        self.decay_factor = (1 - (1 + 3*num_guards_spotted)/10)

    def update_map(self, viewcone: list[list[int]], location: tuple[int, int], direction: int) -> None:
        new_grid = np.array(viewcone, dtype=np.uint8)
        curr_direction = int(direction)
        curr_location = location

        self.num_visited[curr_location[0]][curr_location[1]] += 1

        # Rotate clockwise so absolute north faces up
        new_grid = np.rot90(new_grid, k=curr_direction)

        match curr_direction:
            case Direction.RIGHT: 
                rel_curr_location = (2, 2)
            case Direction.DOWN: 
                rel_curr_location = (2, 2)
            case Direction.LEFT: 
                rel_curr_location = (4, 2) 
            case Direction.UP: 
                rel_curr_location = (2, 4)

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
            true_i = int(curr_location[0] + (i - rel_curr_location[0]))
            if true_i > 15 or true_i < 0: 
                continue 

            for j in range(c):
                true_j = int(curr_location[1] + (j - rel_curr_location[1]))
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
                    opp_i, opp_j = int(true_i + self.DELTA[wall_dir][0]), int(true_j + self.DELTA[wall_dir][1])
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

    def rl(self, observation_json: dict) -> int:
        viewcone = observation_json["viewcone"]
        direction = int(observation_json["direction"])
        location = (int(observation_json["location"][0]), int(observation_json["location"][1]))
        self.step = observation_json["step"]

        if self.step == 0: 
            self.is_scout = bool(observation_json["scout"])
            self.load_rewards()

        self.update_map(viewcone, location, direction)

        current_pos = location
        current_dir = direction

        # Use enhanced navigation with guard avoidance and memory
        action = self.navigation_manager.plan_navigation(current_pos, current_dir)
        
        if action is not None:
            return action
        else:
            # Fallback to random action if no valid navigation plan
            return random.randint(0, 3)
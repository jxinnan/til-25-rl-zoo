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
    
    def add_dangerous_location(self, location: Tuple[int, int], step: int):
        """Add a dangerous location to memory"""
        self.dangerous_locations[location] = step
    
    def is_location_dangerous(self, location: Tuple[int, int], current_step: int) -> bool:
        """Check if a location is considered dangerous"""
        if location not in self.dangerous_locations:
            return False
        
        last_seen = self.dangerous_locations[location]
        return (current_step - last_seen) <= self.memory_duration
    
    def get_dangerous_positions(self, current_step: int) -> Dict[Tuple[int, int], float]:
        """Get all dangerous positions with their danger levels (0.0 to 1.0)"""
        dangerous_positions = {}
        
        for location, last_seen_step in self.dangerous_locations.items():
            steps_ago = current_step - last_seen_step
            if steps_ago <= self.memory_duration:
                # Danger level decreases over time (1.0 = just seen, 0.0 = memory_duration steps ago)
                danger_level = max(0.0, 1.0 - (steps_ago / self.memory_duration))
                dangerous_positions[location] = danger_level
        
        return dangerous_positions
    
    def cleanup_old_memories(self, current_step: int):
        """Remove old dangerous location memories"""
        locations_to_remove = []
        for location, last_seen in self.dangerous_locations.items():
            if (current_step - last_seen) > self.memory_duration:
                locations_to_remove.append(location)
        
        for location in locations_to_remove:
            del self.dangerous_locations[location]

    def get_danger_penalty(self, location: Tuple[int, int], current_step: int) -> float:
        """Get danger penalty for a location (0.0 if safe, higher values for more dangerous)"""
        if location not in self.dangerous_locations:
            return 0.0
        
        last_seen = self.dangerous_locations[location]
        steps_ago = current_step - last_seen
        
        if steps_ago > self.memory_duration:
            return 0.0
        
        # Return penalty that decreases over time
        danger_level = max(0.0, 1.0 - (steps_ago / self.memory_duration))
        return danger_level * 10.0  # Scale to desired penalty range

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
        
        # Calculate minimum required distance from all guards combined
        # If guards are close together, we need to be further away
        if len(self.guard_positions) >= 2:
            min_guard_distance = float('inf')
            for i, guard1 in enumerate(self.guard_positions):
                for j, guard2 in enumerate(self.guard_positions[i+1:], i+1):
                    distance = abs(guard1[0] - guard2[0]) + abs(guard1[1] - guard2[1])
                    min_guard_distance = min(min_guard_distance, distance)
            
            # If guards are very close, increase minimum safe distance
            if min_guard_distance <= 2:
                min_distance = max(min_distance, 5)
                print(f"Guards are close together (distance: {min_guard_distance}), increasing safe distance to {min_distance}")
        
        # Find all positions that are safe
        for x in range(self.size):
            for y in range(self.size):
                is_safe = True
                min_guard_dist = float('inf')
                
                # Check distance from current guards
                for guard_pos in self.guard_positions:
                    distance = abs(x - guard_pos[0]) + abs(y - guard_pos[1])
                    min_guard_dist = min(min_guard_dist, distance)
                    if distance < min_distance:
                        is_safe = False
                        break
                
                # Check if location has dangerous history
                if is_safe and self.guard_memory:
                    if self.guard_memory.is_location_dangerous((x, y), self.current_step):
                        # Require even more distance for historically dangerous locations
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
                    
                    # Bonus for being further from guards
                    guard_distance_bonus = min_guard_dist * 0.1
                    
                    final_score = start_distance + danger_penalty - guard_distance_bonus
                    safe_positions.append((final_score, (x, y)))
        
        if not safe_positions:
            return None
        
        # Sort by score and return best position
        safe_positions.sort()
        selected_position = safe_positions[0][1]
        print(f"Selected safe position: {selected_position} (score: {safe_positions[0][0]:.2f})")
        return selected_position


class GuardAvoidanceManager:
    """Manages guard avoidance using A* pathfinding with memory system and anti-oscillation logic."""
    
    def __init__(self, rl_manager, detection_radius: int = 3, avoidance_radius: int = 2, 
                 memory_duration: int = 100, exit_radius: int = 5, min_avoidance_duration: int = 10):
        self.rl_manager = rl_manager
        self.detection_radius = detection_radius  # Distance at which we start avoiding
        self.avoidance_radius = avoidance_radius  # Minimum safe distance from guards
        self.exit_radius = exit_radius  # Distance required to exit avoidance mode
        self.min_avoidance_duration = min_avoidance_duration  # Minimum steps to stay in avoidance mode
        
        # Guard memory system
        self.guard_memory = GuardMemory(memory_duration)
        
        # Current avoidance state
        self.current_safe_target = None
        self.avoidance_path = []
        self.avoidance_path_index = 0
        self.last_known_guard_positions = []
        
        # Anti-oscillation state
        self.avoidance_start_step = -1
        self.last_guard_detection_step = -1
        self.guard_free_steps = 0  # Count steps without guards
        
        # Performance tracking
        self.avoidance_attempts = 0
        self.successful_avoidances = 0
        
        # Recent position tracking to prevent backtracking
        self.recent_positions = []
        self.position_history_length = 5
    
    def detect_guards(self) -> List[Tuple[int, int]]:
        """Detect current guard positions from RL manager."""
        return self.rl_manager.curr_guard_pos.copy()
    
    def guards_detected(self) -> bool:
        """Check if any guards are currently detected."""
        return len(self.rl_manager.curr_guard_pos) > 0
    
    def is_position_safe(self, pos: Tuple[int, int], buffer_distance: int = None) -> bool:
        """Check if a position is safe from current guards."""
        if buffer_distance is None:
            buffer_distance = self.avoidance_radius
            
        current_guards = self.detect_guards()
        
        for guard_pos in current_guards:
            distance = abs(pos[0] - guard_pos[0]) + abs(pos[1] - guard_pos[1])
            if distance <= buffer_distance:
                return False
        
        return True
    
    def should_avoid_guards(self, current_pos: Tuple[int, int]) -> bool:
        """Determine if guard avoidance should be activated with anti-oscillation logic."""
        current_guards = self.detect_guards()
        current_step = self.rl_manager.step
        
        # Update guard detection tracking
        if current_guards:
            self.last_guard_detection_step = current_step
            self.guard_free_steps = 0
        else:
            self.guard_free_steps += 1
        
        # If no guards, check if we can exit avoidance mode
        if not current_guards:
            # Only exit if we've been guard-free for enough steps
            if self.guard_free_steps >= 10:
                return False
            # If we just started avoidance or haven't been guard-free long enough, continue avoiding
            elif self.avoidance_start_step >= 0:
                return True
            else:
                return False
        
        # Guards detected - check if we should enter/continue avoidance
        min_distance = float('inf')
        for guard_pos in current_guards:
            distance = abs(current_pos[0] - guard_pos[0]) + abs(current_pos[1] - guard_pos[1])
            min_distance = min(min_distance, distance)
        
        # Enter avoidance mode if guard is within detection radius
        if min_distance <= self.detection_radius:
            if self.avoidance_start_step < 0:  # Starting avoidance
                self.avoidance_start_step = current_step
                print(f"Starting avoidance mode at step {current_step}, guard distance: {min_distance}")
            return True
        
        # Continue avoidance if we're already in it and haven't met exit conditions
        if self.avoidance_start_step >= 0:
            steps_in_avoidance = current_step - self.avoidance_start_step
            
            # Must stay in avoidance for minimum duration
            if steps_in_avoidance < self.min_avoidance_duration:
                return True
            
            # Exit only if guards are far enough away
            if min_distance >= self.exit_radius:
                print(f"Exiting avoidance mode at step {current_step}, guard distance: {min_distance}")
                self.avoidance_start_step = -1
                return False
            
            return True
        
        return False
    
    def update_guard_memory(self, current_step: int):
        """Update guard memory with current guard positions."""
        current_guards = self.detect_guards()
        
        # Add current guard positions to memory
        for guard_pos in current_guards:
            self.guard_memory.add_dangerous_location(guard_pos, current_step)
        
        # Clean up old memories
        self.guard_memory.cleanup_old_memories(current_step)
        self.last_known_guard_positions = current_guards
    
    def update_position_history(self, current_pos: Tuple[int, int]):
        """Update recent position history to prevent backtracking"""
        self.recent_positions.append(current_pos)
        if len(self.recent_positions) > self.position_history_length:
            self.recent_positions.pop(0)
    
    def is_position_recently_visited(self, pos: Tuple[int, int]) -> bool:
        """Check if position was recently visited (to prevent backtracking)"""
        return pos in self.recent_positions[-3:] if len(self.recent_positions) >= 3 else False
    
    def find_safe_escape_position(self, current_pos: Tuple[int, int], 
                                current_dir: int) -> Optional[Tuple[int, int]]:
        """Find the optimal safe position to escape to using A*."""
        
        # Create A* planner with current guard positions and memory
        astar = AStar(
            walls=self.rl_manager.obs_wall,
            size=self.rl_manager.size,
            guard_positions=self.detect_guards(),
            avoidance_radius=self.avoidance_radius,
            guard_memory=self.guard_memory,
            current_step=self.rl_manager.step
        )
        
        # Find safe position using A* algorithm with increased minimum distance
        safe_position = astar.find_safe_position(
            start=(current_pos[0], current_pos[1], current_dir),
            min_distance=self.exit_radius  # Use exit radius for better separation
        )
        
        return safe_position
    
    def plan_avoidance_path(self, current_pos: Tuple[int, int], 
                          current_dir: int) -> Optional[List[int]]:
        """Plan optimal path to avoid guards using A*."""
        
        # Find safe escape position
        safe_target = self.find_safe_escape_position(current_pos, current_dir)
        
        if safe_target is None:
            print(f"Warning: No safe escape position found from {current_pos}")
            return None
        
        # Create A* planner with guard avoidance
        astar = AStar(
            walls=self.rl_manager.obs_wall,
            size=self.rl_manager.size,
            guard_positions=self.detect_guards(),
            avoidance_radius=self.avoidance_radius,
            guard_memory=self.guard_memory,
            current_step=self.rl_manager.step
        )
        
        # Plan path to safe position
        path = astar.find_path(
            start=(current_pos[0], current_pos[1], current_dir),
            goal=safe_target
        )
        
        if path is not None:
            print(f"Planned avoidance path from {current_pos} to {safe_target}, length: {len(path)}")
            self.current_safe_target = safe_target
            
        return path
    
    def get_avoidance_action(self, current_pos: Tuple[int, int], 
                           current_dir: int) -> Optional[int]:
        """Get the next action for guard avoidance."""
        
        # Update position history
        self.update_position_history(current_pos)
        
        # Check if we need a new avoidance plan
        need_new_plan = (
            not self.avoidance_path or 
            self.avoidance_path_index >= len(self.avoidance_path) or
            not self.is_current_plan_valid(current_pos, current_dir)
        )
        
        if need_new_plan:
            self.avoidance_attempts += 1
            
            # Plan new avoidance path
            new_path = self.plan_avoidance_path(current_pos, current_dir)
            
            if new_path is None:
                # Emergency fallback: try to move away from closest guard
                return self.get_emergency_action(current_pos, current_dir)
            
            self.avoidance_path = new_path
            self.avoidance_path_index = 0
        
        # Execute current avoidance plan
        if self.avoidance_path and self.avoidance_path_index < len(self.avoidance_path):
            action = self.avoidance_path[self.avoidance_path_index]
            self.avoidance_path_index += 1
            
            # Check if we've completed the avoidance maneuver
            if self.avoidance_path_index >= len(self.avoidance_path):
                if self.is_position_safe(current_pos, self.exit_radius):
                    self.successful_avoidances += 1
                    print(f"Avoidance maneuver completed successfully at {current_pos}")
            
            return action
        
        # Fallback if no action available
        return self.get_emergency_action(current_pos, current_dir)
    
    def is_current_plan_valid(self, current_pos: Tuple[int, int], 
                            current_dir: int) -> bool:
        """Check if the current avoidance plan is still valid."""
        
        if not self.avoidance_path or not self.current_safe_target:
            return False
        
        # Check if target is still safe with exit radius
        if not self.is_position_safe(self.current_safe_target, self.exit_radius):
            return False
        
        # Check if guards have moved significantly
        current_guards = self.detect_guards()
        if len(current_guards) != len(self.last_known_guard_positions):
            return False
        
        # Check for significant guard movement
        guard_moved_significantly = False
        for current_guard in current_guards:
            min_distance = float('inf')
            for last_guard in self.last_known_guard_positions:
                distance = abs(current_guard[0] - last_guard[0]) + abs(current_guard[1] - last_guard[1])
                min_distance = min(min_distance, distance)
            
            if min_distance > 2:  # Guard moved significantly
                guard_moved_significantly = True
                break
        
        return not guard_moved_significantly
    
    def get_emergency_action(self, current_pos: Tuple[int, int], 
                           current_dir: int) -> int:
        """Get emergency action when normal avoidance fails."""
        
        current_guards = self.detect_guards()
        if not current_guards:
            return 0  # Move forward if no guards
        
        # Find direction away from closest guard
        closest_guard = min(current_guards, 
                          key=lambda g: abs(g[0] - current_pos[0]) + abs(g[1] - current_pos[1]))
        
        # Calculate vector away from guard
        away_x = current_pos[0] - closest_guard[0]
        away_y = current_pos[1] - closest_guard[1]
        
        # Determine best direction to move
        direction_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # RIGHT, DOWN, LEFT, UP
        
        best_direction = None
        max_distance_improvement = -1
        
        for i, (dx, dy) in enumerate(direction_vectors):
            # Check if this direction moves us further from the guard
            distance_improvement = away_x * dx + away_y * dy
            
            # Check if movement is valid
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if (0 <= new_pos[0] < self.rl_manager.size and 
                0 <= new_pos[1] < self.rl_manager.size and
                not self.rl_manager.obs_wall[i][current_pos[0]][current_pos[1]] and
                not self.is_position_recently_visited(new_pos)):  # Avoid backtracking
                
                if distance_improvement > max_distance_improvement:
                    max_distance_improvement = distance_improvement
                    best_direction = i
        
        if best_direction is not None:
            # Calculate action needed to face best direction
            if best_direction == current_dir:
                return 0  # Move forward
            elif best_direction == (current_dir + 2) % 4:
                return 1  # Move backward
            elif best_direction == (current_dir + 1) % 4:
                return 3  # Turn right
            else:  # best_direction == (current_dir + 3) % 4
                return 2  # Turn left
        
        # Last resort: turn randomly
        return random.choice([2, 3])  # Turn left or right
    
    def reset_avoidance_state(self):
        """Reset avoidance state when guards are no longer detected."""
        self.current_safe_target = None
        self.avoidance_path = []
        self.avoidance_path_index = 0
        self.avoidance_start_step = -1
        self.guard_free_steps = 0
        self.recent_positions = []
    
    def get_avoidance_statistics(self) -> dict:
        """Get statistics about avoidance performance."""
        success_rate = 0.0
        if self.avoidance_attempts > 0:
            success_rate = self.successful_avoidances / self.avoidance_attempts
        
        return {
            'avoidance_attempts': self.avoidance_attempts,
            'successful_avoidances': self.successful_avoidances,
            'success_rate': success_rate,
            'current_guard_count': len(self.detect_guards()),
            'dangerous_locations_in_memory': len(self.guard_memory.dangerous_locations),
            'is_avoiding': self.avoidance_start_step >= 0,
            'guard_free_steps': self.guard_free_steps,
            'avoidance_duration': self.rl_manager.step - self.avoidance_start_step if self.avoidance_start_step >= 0 else 0
        }
    
    def debug_avoidance_state(self, current_pos: Tuple[int, int]) -> dict:
        """Get detailed debug information about current avoidance state."""
        current_guards = self.detect_guards()
        
        guard_distances = []
        for guard_pos in current_guards:
            distance = abs(current_pos[0] - guard_pos[0]) + abs(current_pos[1] - guard_pos[1])
            guard_distances.append((guard_pos, distance))
        
        return {
            'current_position': current_pos,
            'current_guards': current_guards,
            'guard_distances': guard_distances,
            'should_avoid': self.should_avoid_guards(current_pos),
            'current_safe_target': self.current_safe_target,
            'avoidance_path_length': len(self.avoidance_path),
            'avoidance_progress': f"{self.avoidance_path_index}/{len(self.avoidance_path)}",
            'position_safe': self.is_position_safe(current_pos),
            'dangerous_memory_locations': len(self.guard_memory.get_dangerous_positions(self.rl_manager.step)),
            'guard_free_steps': self.guard_free_steps,
            'avoidance_active': self.avoidance_start_step >= 0
        }

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
    """Manages waypoint-based navigation with guard avoidance and memory."""
    
    def __init__(self, rl_manager):
        self.rl_manager = rl_manager
        self.waypoint_planner = WaypointPlanner(rl_manager)
        self.guard_avoidance = GuardAvoidanceManager(rl_manager)
        self.astar = None
        
        # Current navigation state
        self.current_waypoint = None
        self.current_path = []
        self.path_index = 0
        
        # Guard avoidance state
        self.avoidance_active = False
        self.last_avoidance_check = -1
    
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
        should_avoid = self.guard_avoidance.should_avoid_guards(current_pos)
        
        if should_avoid:
            if not self.avoidance_active:
                # Just started avoiding - clear current path
                print(f"GUARD DETECTED! Activating avoidance from {current_pos}")
                self.clear_current_path()
                self.avoidance_active = True
            
            # Get avoidance action
            action = self.guard_avoidance.get_avoidance_action(current_pos, current_dir)
            return action
        
        else:
            if self.avoidance_active:
                # Just finished avoiding guards
                print(f"Avoidance complete at {current_pos}")
                self.avoidance_active = False
                # Clear current path to force re-planning with updated guard memory
                self.clear_current_path()
        
        # PRIORITY 2: Normal waypoint navigation
        if self.needs_new_waypoint(current_pos):
            # Find best waypoint (considering guard memory)
            self.current_waypoint = self.waypoint_planner.find_best_waypoint(current_pos, current_dir)
            
            if self.current_waypoint is None:
                return None
            
            # Plan path to waypoint using A* with guard memory
            guard_memory = self.guard_avoidance.guard_memory
            
            # Create A* planner with guard awareness
            if hasattr(self, 'create_astar_with_memory'):
                self.astar = self.create_astar_with_memory(guard_memory)
            else:
                # Fallback to basic A* if enhanced version not available
                self.astar = AStar(self.rl_manager.obs_wall, self.rl_manager.size)
            
            self.current_path = self.astar.find_path(
                (current_pos[0], current_pos[1], current_dir), 
                self.current_waypoint
            )
            
            if self.current_path is None:
                # If path planning fails, try a different waypoint
                self.current_waypoint = None
                return self.get_fallback_action(current_pos, current_dir)
            
            self.path_index = 0
            print(f"New path to waypoint {self.current_waypoint}, length: {len(self.current_path)}")
        
        # Follow current path
        if self.current_path and self.path_index < len(self.current_path):
            action = self.current_path[self.path_index]
            self.path_index += 1
            return action
        
        # Fallback if no path available
        return self.get_fallback_action(current_pos, current_dir)
    
    def create_astar_with_memory(self, guard_memory):
        """Create A* planner that considers guard memory"""
        try:
            return AStar(
                self.rl_manager.obs_wall, 
                self.rl_manager.size,
                guard_positions=self.rl_manager.curr_guard_pos,
                guard_memory=guard_memory,
                current_step=self.rl_manager.step
            )
        except TypeError:
            # Fallback if AStar doesn't support these parameters
            return AStar(self.rl_manager.obs_wall, self.rl_manager.size)
    
    def get_fallback_action(self, current_pos: Tuple[int, int], current_dir: int) -> int:
        """Get a fallback action when normal navigation fails"""
        # Try to move forward if safe
        direction_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W
        dx, dy = direction_vectors[current_dir]
        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        # Check if forward movement is safe and valid
        if (0 <= next_pos[0] < self.rl_manager.size and 
            0 <= next_pos[1] < self.rl_manager.size and
            not self.rl_manager.obs_wall[next_pos[0]][next_pos[1]] and
            self.guard_avoidance.is_position_safe(next_pos)):
            return 0  # Move forward
        
        # Try turning right as fallback
        return 1
    
    def get_navigation_status(self) -> dict:
        """Get current navigation status for debugging"""
        return {
            'avoidance_active': self.avoidance_active,
            'current_waypoint': self.current_waypoint,
            'path_length': len(self.current_path) if self.current_path else 0,
            'path_progress': f"{self.path_index}/{len(self.current_path)}" if self.current_path else "0/0",
            'guards_detected': self.guard_avoidance.guards_detected(),
            'guard_count': len(self.guard_avoidance.detect_guards()),
        }
    
    def force_replan(self):
        """Force replanning of current route (useful when environment changes)"""
        self.clear_current_path()
        self.avoidance_active = False
    
    def get_valid_moves(self, current_pos: Tuple[int, int], current_dir: int) -> List[Tuple[int, Tuple[int, int]]]:
        """Get list of valid moves with their resulting positions"""
        valid_moves = []
        
        # Define movement directions: Forward, Right, Backward, Left
        direction_vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W
        
        for action in range(4):  # 0=Forward, 1=Right, 2=Backward, 3=Left
            # Calculate new direction after action
            if action == 0:  # Forward
                new_dir = current_dir
            elif action == 1:  # Turn right
                new_dir = (current_dir + 1) % 4
            elif action == 2:  # Backward
                new_dir = (current_dir + 2) % 4
            else:  # Turn left
                new_dir = (current_dir - 1) % 4
            
            # Calculate new position
            dx, dy = direction_vectors[new_dir]
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check bounds
            if (0 <= new_pos[0] < self.rl_manager.size and 
                0 <= new_pos[1] < self.rl_manager.size):
                # Check for walls (assuming obs_wall exists)
                if hasattr(self.rl_manager, 'obs_wall'):
                    # Use .any() to handle NumPy array boolean evaluation
                    wall_value = self.rl_manager.obs_wall[new_pos[0], new_pos[1]]
                    if not wall_value.any():  # Use .any() for NumPy arrays
                        valid_moves.append((action, new_pos))
                else:
                    # If no wall data available, assume position is valid
                    valid_moves.append((action, new_pos))
        
        return valid_moves

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

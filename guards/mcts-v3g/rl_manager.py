"""Turn-based RL model focused on cornering and trapping scouts."""
import math
import random
from enum import IntEnum
from collections import defaultdict, deque
from typing import Tuple, List, Optional, Set

import numpy as np

from . import gridworld_astar as astar

class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class Tile(IntEnum):
    NO_VISION = 0
    EMPTY = 1
    RECON = 2
    MISSION = 3

class ScoutCorneringAI:
    """AI system specifically designed to corner and trap scouts."""
    
    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        self.scout_history = deque(maxlen=8)
        self.scout_last_seen = -1
        self.current_turn = 0
        self.wall_map = None
        
    def update_walls(self, wall_top, wall_left, wall_bottom, wall_right):
        """Update internal wall map for pathfinding calculations."""
        self.wall_map = {
            'top': wall_top,
            'left': wall_left, 
            'bottom': wall_bottom,
            'right': wall_right
        }
    
    def update_scout_position(self, scout_pos: Optional[Tuple[int, int]]):
        """Update scout tracking."""
        self.current_turn += 1
        if scout_pos:
            self.scout_history.append(scout_pos)
            self.scout_last_seen = self.current_turn
    
    def get_adjacent_positions(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all adjacent positions that are not blocked by walls."""
        x, y = pos
        adjacent = []
        
        if self.wall_map is None:
            # Fallback to all adjacent if no wall info
            for dx, dy in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    adjacent.append((nx, ny))
            return adjacent
        
        # Check each direction for walls
        if y > 0 and self.wall_map['top'][x, y] == 0:  # Up
            adjacent.append((x, y - 1))
        if x > 0 and self.wall_map['left'][x, y] == 0:  # Left
            adjacent.append((x - 1, y))
        if y < self.grid_size - 1 and self.wall_map['bottom'][x, y] == 0:  # Down
            adjacent.append((x, y + 1))
        if x < self.grid_size - 1 and self.wall_map['right'][x, y] == 0:  # Right
            adjacent.append((x + 1, y))
            
        return adjacent
    
    def get_escape_routes(self, scout_pos: Tuple[int, int], player_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Calculate where the scout can escape to from its current position."""
        escape_routes = []
        visited = set()
        queue = deque([(scout_pos, 0)])  # (position, distance)
        
        while queue:
            pos, dist = queue.popleft()
            if pos in visited or dist > 4:  # Don't look too far ahead
                continue
            visited.add(pos)
            
            # Calculate distance from player - farther positions are better escape routes
            player_distance = abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1])
            
            if dist > 0:  # Don't include the scout's current position
                escape_routes.append((pos, player_distance, dist))
            
            # Add adjacent positions to explore
            for adj_pos in self.get_adjacent_positions(pos):
                if adj_pos not in visited:
                    queue.append((adj_pos, dist + 1))
        
        # Sort by player distance (descending) then by scout distance (ascending)
        escape_routes.sort(key=lambda x: (-x[1], x[2]))
        return [pos for pos, _, _ in escape_routes[:8]]  # Return top 8 escape routes
    
    def find_blocking_positions(self, scout_pos: Tuple[int, int], player_pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Find positions that would block the most scout escape routes."""
        escape_routes = self.get_escape_routes(scout_pos, player_pos)
        if not escape_routes:
            return [(scout_pos, 10.0)]  # If no escape routes, go directly to scout
        
        blocking_positions = []
        
        # For each possible blocking position, calculate how many escape routes it cuts off
        search_radius = min(6, max(3, len(escape_routes)))
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                block_pos = (player_pos[0] + dx, player_pos[1] + dy)
                if not (0 <= block_pos[0] < self.grid_size and 0 <= block_pos[1] < self.grid_size):
                    continue
                
                score = self.calculate_blocking_score(block_pos, scout_pos, escape_routes, player_pos)
                if score > 0:
                    blocking_positions.append((block_pos, score))
        
        # Sort by blocking effectiveness
        blocking_positions.sort(key=lambda x: x[1], reverse=True)
        return blocking_positions[:6]  # Return top 6 blocking positions
    
    def calculate_blocking_score(self, block_pos: Tuple[int, int], scout_pos: Tuple[int, int], 
                               escape_routes: List[Tuple[int, int]], player_pos: Tuple[int, int]) -> float:
        """Calculate how effective a position is for blocking scout escapes."""
        score = 0.0
        
        # Distance to scout (closer is better for direct pressure)
        scout_distance = abs(block_pos[0] - scout_pos[0]) + abs(block_pos[1] - scout_pos[1])
        if scout_distance == 1:
            score += 5.0  # Adjacent to scout is very good
        elif scout_distance == 2:
            score += 3.0  # Two steps away is good
        else:
            score += max(0, 2.0 - scout_distance * 0.3)
        
        # Check how many escape routes this position would block
        routes_blocked = 0
        for escape_pos in escape_routes[:5]:  # Check top 5 escape routes
            # If blocking position is on the path between scout and escape route
            if self.is_on_path(scout_pos, escape_pos, block_pos):
                routes_blocked += 1
            # Or if blocking position is adjacent to the escape route
            elif abs(block_pos[0] - escape_pos[0]) + abs(block_pos[1] - escape_pos[1]) <= 1:
                routes_blocked += 0.5
        
        score += routes_blocked * 2.0
        
        # Bonus for positions that create chokepoints
        adjacent_positions = self.get_adjacent_positions(block_pos)
        if len(adjacent_positions) <= 2:  # This is a chokepoint
            score += 1.5
        
        # Distance from current player position (not too far)
        player_distance = abs(block_pos[0] - player_pos[0]) + abs(block_pos[1] - player_pos[1])
        if player_distance <= 3:
            score += 1.0 - player_distance * 0.2
        elif player_distance > 6:
            score -= (player_distance - 6) * 0.5  # Penalty for being too far
        
        return score
    
    def is_on_path(self, start: Tuple[int, int], end: Tuple[int, int], check_pos: Tuple[int, int]) -> bool:
        """Check if a position is roughly on the path between start and end."""
        # Simple line-of-sight check
        if start == end:
            return False
        
        # Check if check_pos is on the straight line between start and end
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        if dx == 0:  # Vertical line
            if check_pos[0] == start[0]:
                return min(start[1], end[1]) <= check_pos[1] <= max(start[1], end[1])
        elif dy == 0:  # Horizontal line
            if check_pos[1] == start[1]:
                return min(start[0], end[0]) <= check_pos[0] <= max(start[0], end[0])
        else:
            # Diagonal or complex path - check if point is close to the line
            cross_product = abs((end[1] - start[1]) * (check_pos[0] - start[0]) - 
                              (end[0] - start[0]) * (check_pos[1] - start[1]))
            line_length = abs(dx) + abs(dy)
            if line_length > 0:
                distance_to_line = cross_product / line_length
                return distance_to_line <= 1.5  # Allow some tolerance
        
        return False
    
    def get_cornering_target(self, player_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get the best position to corner the scout."""
        if not self.scout_history:
            return None
        
        turns_since_seen = self.current_turn - self.scout_last_seen
        if turns_since_seen > 6:  # Scout too old
            return None
        
        last_scout_pos = self.scout_history[-1]
        
        # If scout was seen very recently, be aggressive
        if turns_since_seen <= 1:
            # Try to get adjacent to scout for direct capture
            adjacent_to_scout = self.get_adjacent_positions(last_scout_pos)
            if adjacent_to_scout:
                # Choose the adjacent position closest to player
                best_adjacent = min(adjacent_to_scout, 
                                  key=lambda pos: abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1]))
                return best_adjacent
        
        # Otherwise, find blocking positions
        blocking_positions = self.find_blocking_positions(last_scout_pos, player_pos)
        if blocking_positions:
            return blocking_positions[0][0]  # Return best blocking position
        
        return last_scout_pos  # Fallback to scout's last known position

class RLManager:

    def __init__(self):
        # Initialize model and configurations
        self.size = 16
        
        # Wall observation spaces
        self.obs_wall_top_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_left_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_bottom_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_right_space = np.zeros((self.size, self.size), dtype=np.uint8)
        
        # Scout cornering AI
        self.cornering_ai = ScoutCorneringAI(self.size)
        
        # Exploration tracking
        self.visited_positions = set()
        self.exploration_targets = [
            (8, 8),   # Center
            (4, 4), (12, 4), (4, 12), (12, 12),  # Quarter points
            (2, 2), (14, 14), (2, 14), (14, 2),  # Near corners
            (1, 8), (15, 8), (8, 1), (8, 15),   # Edge midpoints
            (6, 6), (10, 6), (6, 10), (10, 10), # Sub-quarters
        ]
        self.current_exploration_target = 0
        
        # Point tracking
        self.known_points = {}

    def get_next_exploration_target(self, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get the next exploration target."""
        # If current target is reached, move to next
        if self.current_exploration_target < len(self.exploration_targets):
            target = self.exploration_targets[self.current_exploration_target]
            distance_to_target = abs(current_pos[0] - target[0]) + abs(current_pos[1] - target[1])
            if distance_to_target <= 2:
                self.current_exploration_target = (self.current_exploration_target + 1) % len(self.exploration_targets)
        
        # Find the best unvisited exploration target
        best_target = None
        best_score = -1
        
        for i, target in enumerate(self.exploration_targets):
            if target in self.visited_positions:
                continue
                
            distance = abs(current_pos[0] - target[0]) + abs(current_pos[1] - target[1])
            # Prefer closer targets and targets we haven't been to recently
            score = 10.0 - distance * 0.3
            if i == self.current_exploration_target:
                score += 2.0  # Bonus for current target
            
            if score > best_score:
                best_score = score
                best_target = target
        
        if best_target:
            return best_target
        else:
            # All targets visited, pick a random one
            return random.choice(self.exploration_targets)

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation."""
        new_gridview = np.array(observation["viewcone"], dtype=np.uint8)
        curr_direction = np.array(observation["direction"], dtype=np.int64)
        curr_location = np.array(observation["location"], dtype=np.int64)

        # Rotate view to absolute coordinates
        new_gridview = np.rot90(new_gridview, k=curr_direction)

        # Determine relative current location in rotated view
        match curr_direction:
            case Direction.RIGHT: rel_curr_location = (2, 2)
            case Direction.DOWN: rel_curr_location = (2, 2)
            case Direction.LEFT: rel_curr_location = (4, 2)
            case Direction.UP: rel_curr_location = (2, 4)

        # Update world state and look for scouts
        scout_position = None
        
        for i in range(new_gridview.shape[0]):
            new_abs_x = curr_location[0] + i - rel_curr_location[0]
            if new_abs_x < 0 or new_abs_x >= self.size:
                continue
                
            for j in range(new_gridview.shape[1]):
                new_abs_y = curr_location[1] + j - rel_curr_location[1]
                if new_abs_y < 0 or new_abs_y >= self.size:
                    continue

                unpacked = np.unpackbits(new_gridview[i, j])
                tile_contents = np.packbits(np.concatenate((np.zeros(6, dtype=np.uint8), unpacked[-2:])))[0]
                
                if tile_contents != Tile.NO_VISION:
                    # Update wall information
                    wall_bits = list(unpacked[:4])
                    for k in range(curr_direction):
                        wall_bits.append(wall_bits.pop(0))
                    
                    self.obs_wall_top_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[0] * 255)
                    self.obs_wall_left_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[1] * 255)
                    self.obs_wall_bottom_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[2] * 255)
                    self.obs_wall_right_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[3] * 255)
                    
                    # Track points
                    has_recon_point = (tile_contents == Tile.RECON)
                    has_mission_point = (tile_contents == Tile.MISSION)
                    has_any_point = has_recon_point or has_mission_point
                    self.known_points[(new_abs_x, new_abs_y)] = has_any_point

                # Check for scout - THIS IS THE KEY!
                tile_scout_info = unpacked[5]
                if tile_scout_info == 1:
                    scout_position = (new_abs_x, new_abs_y)

        # Update cornering AI with current state
        self.cornering_ai.update_walls(
            self.obs_wall_top_space, self.obs_wall_left_space,
            self.obs_wall_bottom_space, self.obs_wall_right_space
        )
        self.cornering_ai.update_scout_position(scout_position)
        
        # Mark current position as visited
        self.visited_positions.add(tuple(curr_location))

        # Choose target: cornering takes absolute priority
        target_location = self.cornering_ai.get_cornering_target(tuple(curr_location))
        
        if target_location is None:
            # No scout to corner, do exploration
            target_location = self.get_next_exploration_target(tuple(curr_location))

        # Use A* pathfinding to get to target
        ego_loc = list(observation["location"])
        ego_dir = observation["direction"]
        
        if ego_dir == Direction.LEFT or ego_dir == Direction.RIGHT:
            ego_loc[0] += 16
        ego_loc = tuple(ego_loc)

        astar_half_grid = np.dstack([
            self.obs_wall_top_space, 
            self.obs_wall_left_space, 
            self.obs_wall_bottom_space, 
            self.obs_wall_right_space
        ])
        astar_grid = np.tile(astar_half_grid, (2, 1, 1))
        astar_path = astar.find_path(astar_grid, ego_loc, target_location)
        
        try:
            next_loc = astar_path[1]
        except:
            next_loc = ego_loc
            
        try:
            next_next_loc = astar_path[2]
        except:
            next_next_loc = next_loc

        # Convert path to action
        action = random.randint(0, 3)  # Default random action
        
        if next_loc[0] == ego_loc[0] + 16 or next_loc[0] == ego_loc[0] - 16:  # Dimension change
            # Look ahead to next move for better orientation
            if next_next_loc[1] == next_loc[1] - 1:  # Move up
                if ego_dir == Direction.LEFT: action = 3  # Turn right
                elif ego_dir == Direction.RIGHT: action = 2  # Turn left
            elif next_next_loc[1] == next_loc[1] + 1:  # Move down
                if ego_dir == Direction.LEFT: action = 2  # Turn left
                elif ego_dir == Direction.RIGHT: action = 3  # Turn right
            elif next_next_loc[0] == next_loc[0] - 1:  # Move left
                if ego_dir == Direction.UP: action = 2  # Turn left
                elif ego_dir == Direction.DOWN: action = 3  # Turn right
            elif next_next_loc[0] == next_loc[0] + 1:  # Move right
                if ego_dir == Direction.UP: action = 3  # Turn right
                elif ego_dir == Direction.DOWN: action = 2  # Turn left
        elif next_loc[1] == ego_loc[1] - 1:  # Move up
            if ego_dir == Direction.UP: action = 0  # Move forward
            elif ego_dir == Direction.DOWN: action = 1  # Move backward
        elif next_loc[1] == ego_loc[1] + 1:  # Move down
            if ego_dir == Direction.UP: action = 1  # Move backward
            elif ego_dir == Direction.DOWN: action = 0  # Move forward
        elif next_loc[0] == ego_loc[0] - 1:  # Move left
            if ego_dir == Direction.LEFT: action = 0  # Move forward
            elif ego_dir == Direction.RIGHT: action = 1  # Move backward
        elif next_loc[0] == ego_loc[0] + 1:  # Move right
            if ego_dir == Direction.LEFT: action = 1  # Move backward
            elif ego_dir == Direction.RIGHT: action = 0  # Move forward

        return action
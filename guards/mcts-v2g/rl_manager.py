"""Manages the RL model with MCTS for open room prioritization."""
import math
import random
from enum import IntEnum
from collections import defaultdict
from typing import Tuple, List, Optional

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

class MCTSNode:
    def __init__(self, position: Tuple[int, int], parent=None):
        self.position = position
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]
    
    def expand(self, action):
        child = MCTSNode(action, parent=self)
        self.untried_actions.remove(action)
        self.children.append(child)
        return child
    
    def update(self, reward):
        self.visits += 1
        self.value += reward

class RLManager:

    def __init__(self):
        # Initialize model and configurations
        self.size = 16
        
        # Wall observation spaces
        self.obs_wall_top_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_left_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_bottom_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_right_space = np.zeros((self.size, self.size), dtype=np.uint8)
        
        # MCTS parameters
        self.mcts_iterations = 50
        self.exploration_bonus = 2.0
        
        # Scout tracking
        self.last_scout_location = None
        self.last_scout_turn = 0
        self.scout_chase_duration = 5  # Chase for 5 turns after seeing scout
        
        # Room openness cache
        self.room_openness_cache = {}
        self.visited_positions = set()
        
        # Scout tracking and breadcrumbs
        self.scout_origin = (0, 0)
        self.collected_points = set()  # Positions where points were collected
        self.breadcrumb_trail = []     # Ordered list of collected points (most recent first)
        self.max_breadcrumbs = 10      # Keep track of last 10 collected points
        
        # Point tracking
        self.known_points = {}         # Maps position -> True if point exists, False if collected
        self.point_collection_order = []  # Order in which points were collected
        
        # High-value positions (open areas, strategic locations)
        self.strategic_positions = [
            (8, 8),   # Center
            (4, 4),   # Quarter points
            (12, 4),
            (4, 12),
            (12, 12),
            (2, 2),   # Near scout origin
            (1, 3),
            (3, 1)
        ]

    def update_point_tracking(self, position: Tuple[int, int], has_point: bool):
        """Update our knowledge of points at a position."""
        if position not in self.known_points:
            self.known_points[position] = has_point
        elif self.known_points[position] and not has_point:
            # Point was collected since we last saw it!
            self.collected_points.add(position)
            self.breadcrumb_trail.insert(0, position)  # Add to front (most recent)
            self.point_collection_order.append(position)
            
            # Limit breadcrumb trail size
            if len(self.breadcrumb_trail) > self.max_breadcrumbs:
                self.breadcrumb_trail.pop()
            
            # Update known state
            self.known_points[position] = False

    def get_breadcrumb_score(self, position: Tuple[int, int]) -> float:
        """Calculate breadcrumb-based priority for a position."""
        if position in self.collected_points:
            # Direct hit - this position had a collected point
            recency_index = self.breadcrumb_trail.index(position) if position in self.breadcrumb_trail else len(self.breadcrumb_trail)
            recency_bonus = max(0, 3.0 - recency_index * 0.3)  # More recent = higher bonus
            return 4.0 + recency_bonus
        
        # Check proximity to breadcrumbs (scout likely passed through nearby areas)
        min_distance = float('inf')
        closest_breadcrumb_age = 0
        
        for i, breadcrumb in enumerate(self.breadcrumb_trail):
            distance = abs(position[0] - breadcrumb[0]) + abs(position[1] - breadcrumb[1])
            if distance < min_distance:
                min_distance = distance
                closest_breadcrumb_age = i
        
        if min_distance <= 2:  # Within 2 tiles of a breadcrumb
            proximity_bonus = max(0, 2.0 - min_distance * 0.5)
            age_bonus = max(0, 1.0 - closest_breadcrumb_age * 0.1)
            return proximity_bonus + age_bonus
        
        return 0.0

    def predict_scout_direction(self) -> Optional[Tuple[int, int]]:
        """Predict where the scout might be heading based on breadcrumb pattern."""
        if len(self.breadcrumb_trail) < 2:
            return None
        
        # Look at the most recent movement pattern
        recent_trail = self.breadcrumb_trail[:3]  # Last 3 collected points
        
        # Calculate average direction vector
        direction_vectors = []
        for i in range(len(recent_trail) - 1):
            current = recent_trail[i]
            previous = recent_trail[i + 1]
            dx = current[0] - previous[0]
            dy = current[1] - previous[1]
            direction_vectors.append((dx, dy))
        
        if not direction_vectors:
            return None
        
        # Average the direction vectors
        avg_dx = sum(dx for dx, dy in direction_vectors) / len(direction_vectors)
        avg_dy = sum(dy for dx, dy in direction_vectors) / len(direction_vectors)
        
        # Project forward from most recent breadcrumb
        most_recent = self.breadcrumb_trail[0]
        predicted_x = int(most_recent[0] + avg_dx * 2)  # Project 2 steps ahead
        predicted_y = int(most_recent[1] + avg_dy * 2)
        
        # Clamp to grid bounds
        predicted_x = max(0, min(self.size - 1, predicted_x))
        predicted_y = max(0, min(self.size - 1, predicted_y))
        
        return (predicted_x, predicted_y)
        """Calculate how 'open' a room is based on wall density around it."""
        if position in self.room_openness_cache:
            return self.room_openness_cache[position]
        
        x, y = position
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return 0.0
        
        # Count walls in a 3x3 area around the position
        wall_count = 0
        total_checks = 0
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    # Count walls around this cell
                    wall_count += (self.obs_wall_top_space[nx, ny] > 0)
                    wall_count += (self.obs_wall_left_space[nx, ny] > 0)
                    wall_count += (self.obs_wall_bottom_space[nx, ny] > 0)
                    wall_count += (self.obs_wall_right_space[nx, ny] > 0)
                    total_checks += 4
        
        if total_checks == 0:
            openness = 1.0  # Assume open if no data
        else:
            openness = 1.0 - (wall_count / total_checks)
        
        self.room_openness_cache[position] = openness
        return openness

    def get_valid_moves(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid adjacent moves from a position."""
        x, y = position
        moves = []
        
        # Check each direction
        if y > 0 and self.obs_wall_top_space[x, y] == 0:  # Up
            moves.append((x, y - 1))
        if x > 0 and self.obs_wall_left_space[x, y] == 0:  # Left
            moves.append((x - 1, y))
        if y < self.size - 1 and self.obs_wall_bottom_space[x, y] == 0:  # Down
            moves.append((x, y + 1))
        if x < self.size - 1 and self.obs_wall_right_space[x, y] == 0:  # Right
            moves.append((x + 1, y))
        
        return moves

    def calculate_room_openness(self, position: Tuple[int, int]) -> float:
        """Calculate how 'open' a room is based on wall density around it."""
        if position in self.room_openness_cache:
            return self.room_openness_cache[position]
        
        x, y = position
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return 0.0
        
        # Count walls in a 3x3 area around the position
        wall_count = 0
        total_checks = 0
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    # Count walls around this cell
                    wall_count += (self.obs_wall_top_space[nx, ny] > 0)
                    wall_count += (self.obs_wall_left_space[nx, ny] > 0)
                    wall_count += (self.obs_wall_bottom_space[nx, ny] > 0)
                    wall_count += (self.obs_wall_right_space[nx, ny] > 0)
                    total_checks += 4
        
        if total_checks == 0:
            openness = 1.0  # Assume open if no data
        else:
            openness = 1.0 - (wall_count / total_checks)
        
        self.room_openness_cache[position] = openness
        return openness

    def evaluate_position(self, position: Tuple[int, int], current_pos: Tuple[int, int]) -> float:
        """Evaluate the value of a position for MCTS."""
        # Base score from room openness
        openness_score = self.calculate_room_openness(position) * 2.0
        
        # Breadcrumb-based scoring (highest priority)
        breadcrumb_score = self.get_breadcrumb_score(position) * 2.0
        
        # Bonus for unvisited positions
        exploration_bonus = 1.5 if position not in self.visited_positions else 0.0
        
        # Strategic position bonus
        strategic_bonus = 1.0 if position in self.strategic_positions else 0.0
        
        # Distance from scout origin (closer is better for interception)
        origin_distance = abs(position[0] - self.scout_origin[0]) + abs(position[1] - self.scout_origin[1])
        origin_bonus = max(0, 1.5 - origin_distance * 0.1)
        
        # Predicted scout location bonus
        predicted_location = self.predict_scout_direction()
        prediction_bonus = 0.0
        if predicted_location:
            pred_distance = abs(position[0] - predicted_location[0]) + abs(position[1] - predicted_location[1])
            prediction_bonus = max(0, 2.0 - pred_distance * 0.3)
        
        # Penalty for being too close to current position (encourage movement)
        current_distance = abs(position[0] - current_pos[0]) + abs(position[1] - current_pos[1])
        movement_bonus = min(current_distance * 0.4, 0.8)
        
        return (breadcrumb_score + openness_score + exploration_bonus + 
                strategic_bonus + origin_bonus + prediction_bonus + movement_bonus)

    def mcts_search(self, root_position: Tuple[int, int], depth: int = 3) -> Tuple[int, int]:
        """Perform MCTS to find the best next position."""
        root = MCTSNode(root_position)
        root.untried_actions = self.get_valid_moves(root_position)
        
        for _ in range(self.mcts_iterations):
            # Selection
            node = root
            path = [node]
            
            while node.untried_actions == [] and node.children != []:
                node = node.best_child()
                path.append(node)
            
            # Expansion
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                node = node.expand(action)
                path.append(node)
            
            # Simulation
            current_pos = node.position
            total_reward = 0
            
            for step in range(depth):
                possible_moves = self.get_valid_moves(current_pos)
                if not possible_moves:
                    break
                
                # Choose move based on evaluation
                move_scores = [(move, self.evaluate_position(move, root_position)) 
                              for move in possible_moves]
                move_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Add some randomness to simulation
                if len(move_scores) > 1 and random.random() < 0.3:
                    current_pos = move_scores[1][0]  # Sometimes pick second best
                else:
                    current_pos = move_scores[0][0]
                
                total_reward += self.evaluate_position(current_pos, root_position)
            
            # Backpropagation
            for node in path:
                node.update(total_reward)
        
        if root.children:
            best_child = root.best_child(c_param=0)  # Exploit only
            return best_child.position
        elif root.untried_actions:
            return random.choice(root.untried_actions)
        else:
            return root_position

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
        scout_spotted = False
        scout_location = None
        
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
                    
                    # Track points for breadcrumb detection
                    has_recon_point = (tile_contents == Tile.RECON)
                    has_mission_point = (tile_contents == Tile.MISSION)
                    has_any_point = has_recon_point or has_mission_point
                    
                    self.update_point_tracking((new_abs_x, new_abs_y), has_any_point)

                # Check for scout
                tile_scout_info = unpacked[5]
                if tile_scout_info == 1:
                    scout_spotted = True
                    scout_location = (new_abs_x, new_abs_y)
                    self.last_scout_location = scout_location
                    self.last_scout_turn = 0

        # Mark current position as visited
        self.visited_positions.add(tuple(curr_location))
        
        # Increment scout chase timer
        self.last_scout_turn += 1

        # Determine target location
        if scout_spotted or (self.last_scout_location and self.last_scout_turn < self.scout_chase_duration):
            # Chase mode: go after scout (highest priority)
            target_location = scout_location if scout_spotted else self.last_scout_location
        elif self.breadcrumb_trail:
            # Breadcrumb mode: follow the trail or predict scout location
            predicted_location = self.predict_scout_direction()
            if predicted_location and self.last_scout_turn < self.scout_chase_duration * 2:
                # Chase predicted location if recent trail exists
                target_location = predicted_location
            else:
                # Head to most recent breadcrumb area
                target_location = self.mcts_search(tuple(curr_location))
        else:
            # Exploration mode: use MCTS to find best position
            target_location = self.mcts_search(tuple(curr_location))

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
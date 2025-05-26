"""Manages the RL model."""
from enum import IntEnum
from random import randint

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

class RLManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        
        self.size = 16

        self.obs_wall_top_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_left_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_bottom_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_right_space = np.zeros((self.size, self.size), dtype=np.uint8)

        # --- GUARDS ---
        self.search_dsts = [(3,3), (self.size-4, self.size-4), (self.size-4, 3), (3, self.size-4)]
        
        # --- SWISS Cheese ---
        self.last_scout = None
        self.last_scout_turn = 4

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
        new_gridview = np.array(observation["viewcone"], dtype=np.uint8)
        curr_direction = np.array(observation["direction"], dtype=np.int64) # right down left up
        curr_location = np.array(observation["location"], dtype=np.int64)

        # rotate clockwise so absolute north faces up
        new_gridview = np.rot90(new_gridview, k=curr_direction)

        match curr_direction: # location of self in rotated new_gridview
            case Direction.RIGHT: rel_curr_location = (2,2)
            case Direction.DOWN: rel_curr_location = (2,2)
            case Direction.LEFT: rel_curr_location = (4,2)
            case Direction.UP: rel_curr_location = (2,4)

        # update tile by tile, column by column, in global POV
        if np.array_equal(curr_location, np.array(self.search_dsts[0])):
            self.search_dsts.append(self.search_dsts.pop(0))
        scout_loc = self.search_dsts[0]
        
        # --- SWISS Cheese ---
        self.last_scout_turn += 1
        if self.last_scout_turn < 3:
            scout_loc = self.last_scout
        
        for i in range(new_gridview.shape[0]):
            new_abs_x = curr_location[0] + i - rel_curr_location[0]
            if new_abs_x < 0 or new_abs_x >= self.size: continue
            for j in range(new_gridview.shape[1]):
                new_abs_y = curr_location[1] + j - rel_curr_location[1]
                if new_abs_y < 0 or new_abs_y >= self.size: continue

                # extract data
                unpacked = np.unpackbits(new_gridview[i, j])

                # update last seen and rewards
                tile_contents = np.packbits(np.concatenate((np.zeros(6, dtype=np.uint8), unpacked[-2:])))[0]
                if tile_contents != Tile.NO_VISION:
                    # store wall
                    # wall is given as relative to agent frame, where agent always faces right
                    # given as top left bottom right
                    wall_bits = list(unpacked[:4])
                    # rotate clockwise
                    for k in range(curr_direction): # direction 0-3 right down left up
                        wall_bits.append(wall_bits.pop(0))
                    self.obs_wall_top_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[0] * 255)
                    self.obs_wall_left_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[1] * 255)
                    self.obs_wall_bottom_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[2] * 255)
                    self.obs_wall_right_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[3] * 255)

                # update visible guards
                tile_scout_info = unpacked[5]
                if tile_scout_info == 1:
                    scout_loc = (new_abs_x, new_abs_y)
                    
                    # --- SWISS Cheese ---
                    self.last_scout = (new_abs_x, new_abs_y)
                    self.last_scout_turn = 0
        
        ego_loc = list(observation["location"])
        ego_dir = observation["direction"]
        if ego_dir == Direction.LEFT or ego_dir == Direction.RIGHT: ego_loc[0] += 16
        ego_loc = tuple(ego_loc)

        astar_half_grid = np.dstack([self.obs_wall_top_space, self.obs_wall_left_space, self.obs_wall_bottom_space, self.obs_wall_right_space])
        astar_grid = np.tile(astar_half_grid, (2,1,1))
        astar_path = astar.find_path(astar_grid, ego_loc, scout_loc)
        try:
            next_loc = astar_path[1]
        except:
            next_loc = ego_loc
            
        # --- LATIN AMERICAN Cheese ---
        try:
            next_next_loc = astar_path[2]
        except:
            next_next_loc = next_loc
            
        action = randint(0,3)
        if next_loc[0] == ego_loc[0] + 16 or next_loc[0] == ego_loc[0] - 16: # change dimension
            # --- LATIN AMERICAN Cheese ---
            if next_next_loc[1] == next_loc[1] - 1: # move up
                if ego_dir == Direction.LEFT: action = 3 # turn right
                elif ego_dir == Direction.RIGHT: action = 2 # turn left
            elif next_next_loc[1] == next_loc[1] + 1: # move down
                if ego_dir == Direction.LEFT: action = 2 # turn left
                elif ego_dir == Direction.RIGHT: action = 3 # turn right
            elif next_next_loc[0] == next_loc[0] - 1: # move left
                if ego_dir == Direction.UP: action = 2 # turn left
                elif ego_dir == Direction.DOWN: action = 3 # turn right
            elif next_next_loc[0] == next_loc[0] + 1: # move right
                if ego_dir == Direction.UP: action = 3 # turn right
                elif ego_dir == Direction.DOWN: action = 2 # turn left
            
            # action = 2 # turn left
        elif next_loc[1] == ego_loc[1] - 1: # move up
            if ego_dir == Direction.UP: action = 0 # move forward
            elif ego_dir == Direction.DOWN: action = 1 # move backward
        elif next_loc[1] == ego_loc[1] + 1: # move down
            if ego_dir == Direction.UP: action = 1 # move backward
            elif ego_dir == Direction.DOWN: action = 0 # move forward
        elif next_loc[0] == ego_loc[0] - 1: # move left
            if ego_dir == Direction.LEFT: action = 0 # move forward
            elif ego_dir == Direction.RIGHT: action = 1 # move backward
        elif next_loc[0] == ego_loc[0] + 1: # move right
            if ego_dir == Direction.LEFT: action = 1 # move backward
            elif ego_dir == Direction.RIGHT: action = 0 # move forward

        return action
            
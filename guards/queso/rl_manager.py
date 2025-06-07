"""Manages the RL model."""
from enum import IntEnum
from random import randint

import numpy as np

from . import gridworld_astar as astar

import cv2

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
    TURN_PROB = 0.5
    REVERSE_PROB = 0.1
    FORWARD_PROB = 1 - TURN_PROB - REVERSE_PROB
    ROUND_COUNT = 101

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        
        self.size = 16

        """
        32x16 grid
        [:16,:] represents facing up/down (moving vertically)
        [16:,:] represents facing left/right (moving horizontally)
        """
        # up/left (negative), down/right (positive), turn
        self.scout_prob = np.zeros((self.ROUND_COUNT, 2 * self.size, self.size, 3), dtype=np.float32)
        self.repeated_prob = np.full((self.ROUND_COUNT, self.size, self.size), 0, dtype=np.float32)

        self.obs_wall_top_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_left_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_bottom_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_right_space = np.zeros((self.size, self.size), dtype=np.uint8)

        self.faux_wall_top_space = np.zeros((self.ROUND_COUNT, self.size, self.size), dtype=np.uint8)
        self.faux_wall_left_space = np.zeros((self.ROUND_COUNT, self.size, self.size), dtype=np.uint8)
        self.faux_wall_bottom_space = np.zeros((self.ROUND_COUNT, self.size, self.size), dtype=np.uint8)
        self.faux_wall_right_space = np.zeros((self.ROUND_COUNT, self.size, self.size), dtype=np.uint8)

        self.obs_wall_top_space[:, 0] = 255
        self.obs_wall_left_space[0, :] = 255
        self.obs_wall_bottom_space[:, -1] = 255
        self.obs_wall_right_space[-1, :] = 255

        self.curr_turn = 0
        self.last_seen_turn = 0

    def calc_next_repeated(self, turn_idx, x_idx, y_idx):
        prev_turn_idx = turn_idx - 1
        if prev_turn_idx < 0:
            prev_prob = 1
        else:
            prev_prob = 1 - self.repeated_prob[prev_turn_idx, x_idx, y_idx]

        curr_prob = prev_prob * (1 - \
            np.sum(self.scout_prob[turn_idx, x_idx, y_idx]) - \
            np.sum(self.scout_prob[turn_idx, x_idx+16, y_idx]))
        
        self.repeated_prob[turn_idx, x_idx, y_idx] = 1 - curr_prob

    def calc_next(self, turn_idx, x_idx, y_idx):
        prev_turn_idx = turn_idx - 1
        if x_idx < 16: # vertical
            turn_x_idx = x_idx + 16
            true_neg_x_idx = x_idx
            neg_x_idx = x_idx
            neg_y_idx = y_idx - 1
            true_pos_x_idx = x_idx
            pos_x_idx = x_idx
            pos_y_idx = y_idx + 1
            aft_turn_x_idx_1 = x_idx - 1
            aft_turn_y_idx_1 = y_idx
            aft_turn_x_idx_2 = x_idx + 1
            aft_turn_y_idx_2 = y_idx
            can_neg = False if self.obs_wall_top_space[x_idx, y_idx] == 255 else True
            can_pos = False if self.obs_wall_bottom_space[x_idx, y_idx] == 255 else True
            can_turn_1 = False if self.obs_wall_left_space[x_idx, y_idx] == 255 else True
            can_turn_2 = False if self.obs_wall_right_space[x_idx, y_idx] == 255 else True
            can_turn = (can_turn_1 or can_turn_2)
        else: # horizontal
            turn_x_idx = x_idx - 16
            true_neg_x_idx = turn_x_idx - 1
            neg_x_idx = x_idx - 1
            neg_y_idx = y_idx
            true_pos_x_idx = turn_x_idx + 1
            pos_x_idx = x_idx + 1
            pos_y_idx = y_idx
            aft_turn_x_idx_1 = turn_x_idx
            aft_turn_y_idx_1 = y_idx - 1
            aft_turn_x_idx_2 = turn_x_idx
            aft_turn_y_idx_2 = y_idx + 1
            can_neg = False if self.obs_wall_left_space[turn_x_idx, y_idx] == 255 else True
            can_pos = False if self.obs_wall_right_space[turn_x_idx, y_idx] == 255 else True
            can_turn_1 = False if self.obs_wall_top_space[turn_x_idx, y_idx] == 255 else True
            can_turn_2 = False if self.obs_wall_bottom_space[turn_x_idx, y_idx] == 255 else True
            can_turn = (can_turn_1 or can_turn_2)
        
        # check if any possible moves exist, and if on first turn
        if not (can_neg or can_pos or can_turn) or prev_turn_idx < 0:
            return
        
        neg_repeated = 0
        pos_repeated = 0
        turn_repeated = 0

        if can_neg:
            neg_repeated = np.maximum(self.REVERSE_PROB, (1 - self.repeated_prob[prev_turn_idx, true_neg_x_idx, neg_y_idx]))
        if can_pos:
            pos_repeated = np.maximum(self.REVERSE_PROB, (1 - self.repeated_prob[prev_turn_idx, true_pos_x_idx, pos_y_idx]))
        if can_turn_1:
            turn_repeated += 0.5 * np.maximum(self.REVERSE_PROB, (1 - self.repeated_prob[prev_turn_idx, aft_turn_x_idx_1, aft_turn_y_idx_1]))
        if can_turn_2:
            turn_repeated += 0.5 * np.maximum(self.REVERSE_PROB, (1 - self.repeated_prob[prev_turn_idx, aft_turn_x_idx_2, aft_turn_y_idx_2]))

        # process input from negative tile
        if can_neg and neg_y_idx >= 0 and ( \
            (x_idx < self.size and neg_x_idx >= 0) \
            or (x_idx >= self.size and neg_x_idx >= self.size) \
        ): # within map
            fr_neg_prob = self.scout_prob[prev_turn_idx, neg_x_idx, neg_y_idx, 1] # positive prob from negative tile
            
            total_prob = \
                self.REVERSE_PROB * neg_repeated + \
                self.FORWARD_PROB * pos_repeated + \
                self.TURN_PROB * turn_repeated
            
            if can_neg: self.scout_prob[turn_idx, x_idx, y_idx, 0] += self.REVERSE_PROB * neg_repeated / total_prob * fr_neg_prob
            if can_pos: self.scout_prob[turn_idx, x_idx, y_idx, 1] += self.FORWARD_PROB * pos_repeated / total_prob * fr_neg_prob
            if can_turn: self.scout_prob[turn_idx, x_idx, y_idx, 2] += self.TURN_PROB * turn_repeated / total_prob * fr_neg_prob

        # process input from positive tile
        if can_pos and pos_y_idx < self.size and ( \
            (x_idx < 16 and pos_x_idx < self.size) \
            or (x_idx >= 16 and pos_x_idx < 2 * self.size) \
        ): # within map
            fr_pos_prob = self.scout_prob[prev_turn_idx, pos_x_idx, pos_y_idx, 0] # negative prob from positive tile
            
            total_prob = \
                self.FORWARD_PROB * neg_repeated + \
                self.REVERSE_PROB * pos_repeated + \
                self.TURN_PROB * turn_repeated
            
            if can_neg: self.scout_prob[turn_idx, x_idx, y_idx, 0] += self.FORWARD_PROB * neg_repeated / total_prob * fr_pos_prob
            if can_pos: self.scout_prob[turn_idx, x_idx, y_idx, 1] += self.REVERSE_PROB * pos_repeated / total_prob * fr_pos_prob
            if can_turn: self.scout_prob[turn_idx, x_idx, y_idx, 2] += self.TURN_PROB * turn_repeated / total_prob * fr_pos_prob

        # process input from turning
        if can_turn:
            fr_turn_prob = self.scout_prob[prev_turn_idx, turn_x_idx, y_idx, 2] # turn prob from other dimension
            
            total_prob = neg_repeated + pos_repeated # won't turn back again

            if total_prob > 0:
                if can_neg: self.scout_prob[turn_idx, x_idx, y_idx, 0] += neg_repeated / total_prob * fr_turn_prob
                if can_pos: self.scout_prob[turn_idx, x_idx, y_idx, 1] += pos_repeated / total_prob * fr_turn_prob
    
    def shape_obs(self, observation):
        scout_loc = (-1, -1)
        walls_changed = False

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
                    if self.obs_wall_top_space[new_abs_x, new_abs_y] != wall_bits[0] * 255:
                        walls_changed = True
                        self.obs_wall_top_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[0] * 255)
                    if self.obs_wall_left_space[new_abs_x, new_abs_y] != wall_bits[1] * 255:
                        walls_changed = True
                        self.obs_wall_left_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[1] * 255)
                    if self.obs_wall_bottom_space[new_abs_x, new_abs_y] != wall_bits[2] * 255:
                        walls_changed = True
                        self.obs_wall_bottom_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[2] * 255)
                    if self.obs_wall_right_space[new_abs_x, new_abs_y] != wall_bits[3] * 255:
                        walls_changed = True
                        self.obs_wall_right_space[new_abs_x, new_abs_y] = np.uint8(wall_bits[3] * 255)

                    # we know the wall on the other side too
                    if new_abs_y + 1 < self.size: # top wall of the tile below
                        self.obs_wall_top_space[new_abs_x, new_abs_y+1] = np.uint8(wall_bits[2] * 255)
                    if new_abs_x + 1 < self.size: # left wall of the tile to the right
                        self.obs_wall_left_space[new_abs_x+1, new_abs_y] = np.uint8(wall_bits[3] * 255)
                    if new_abs_y - 1 >= 0: # bottom wall of the tile above
                        self.obs_wall_bottom_space[new_abs_x, new_abs_y-1] = np.uint8(wall_bits[0] * 255)
                    if new_abs_x - 1 >= 0: # right wall of the tile to the left
                        self.obs_wall_right_space[new_abs_x-1, new_abs_y] = np.uint8(wall_bits[1] * 255)

                # update visible guards
                tile_scout_info = unpacked[5]
                if tile_scout_info == 1:
                    scout_loc = (new_abs_x, new_abs_y)
        return scout_loc, walls_changed
    
    def seen_scout(self, scout_loc):
        self.last_seen_turn = self.curr_turn
        self.scout_prob[self.curr_turn, :, :, :] = 0

        turn_idx = self.curr_turn
        x_idx = scout_loc[0]
        y_idx = scout_loc[1]

        prev_turn_idx = turn_idx - 1

        turn_x_idx = x_idx + 16
        v_neg_x_idx = x_idx
        v_neg_y_idx = y_idx - 1
        v_pos_x_idx = x_idx
        v_pos_y_idx = y_idx + 1
        v_can_neg = False if self.obs_wall_top_space[x_idx, y_idx] == 255 else True
        v_can_pos = False if self.obs_wall_bottom_space[x_idx, y_idx] == 255 else True
        v_can_exist = False if self.obs_wall_top_space[x_idx, y_idx] == 255 and self.obs_wall_bottom_space[x_idx, y_idx] == 255 else True
        
        h_neg_x_idx = turn_x_idx - 1
        h_neg_y_idx = y_idx
        h_pos_x_idx = turn_x_idx + 1
        h_pos_y_idx = y_idx
        h_can_neg = False if self.obs_wall_left_space[x_idx, y_idx] == 255 else True
        h_can_pos = False if self.obs_wall_right_space[x_idx, y_idx] == 255 else True
        h_can_exist = False if self.obs_wall_left_space[x_idx, y_idx] == 255 and self.obs_wall_right_space[x_idx, y_idx] == 255 else True

        move_one_dir_prob = (self.REVERSE_PROB + self.FORWARD_PROB)/2
        # process vertical direction
        if v_can_exist:
            total_dir_prob = v_can_exist / (v_can_exist + h_can_exist)
            
            total_prob = \
                (v_can_neg + v_can_pos) * move_one_dir_prob + \
                h_can_exist * self.TURN_PROB
            
            if v_can_neg: self.scout_prob[turn_idx, x_idx, y_idx, 0] += move_one_dir_prob/total_prob * total_dir_prob
            if v_can_pos: self.scout_prob[turn_idx, x_idx, y_idx, 1] += move_one_dir_prob/total_prob * total_dir_prob
            if h_can_exist: self.scout_prob[turn_idx, x_idx, y_idx, 2] += self.TURN_PROB/total_prob * total_dir_prob
        
        # process horizontal direction
        if h_can_exist:
            total_dir_prob = h_can_exist / (v_can_exist + h_can_exist)
            
            total_prob = \
                (h_can_neg + h_can_pos) * move_one_dir_prob + \
                v_can_exist * self.TURN_PROB
            
            if h_can_neg: self.scout_prob[turn_idx, turn_x_idx, y_idx, 0] += move_one_dir_prob/total_prob * total_dir_prob
            if h_can_pos: self.scout_prob[turn_idx, turn_x_idx, y_idx, 1] += move_one_dir_prob/total_prob * total_dir_prob
            if v_can_exist: self.scout_prob[turn_idx, turn_x_idx, y_idx, 2] += self.TURN_PROB/total_prob * total_dir_prob

    def visualise(self):
        if self.curr_turn == 0:
            out_img = np.full((640,640), 0, dtype=np.float32)
            out_img = cv2.resize(out_img, (640,640), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Scout Probability", out_img)
            cv2.waitKey(10)
            cv2.imshow("Scout Probability", out_img)
            cv2.waitKey(10)
            # from time import sleep
            # sleep(5)

        # out_img = self.repeated_prob[self.curr_turn]
        out_img = np.sum(self.scout_prob[self.curr_turn, :16], axis=2) + np.sum(self.scout_prob[self.curr_turn, 16:], axis=2)
        # print(np.sum(out_img))
        out_img *= 1/np.max(out_img)
        out_img = cv2.resize(out_img, (640,640), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Scout Probability", out_img)
        cv2.waitKey(100)

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation.

        Args:
            observation: The observation from the environment. See
                `rl/README.md` for the format.

        Returns:
            An integer representing the action to take. See `rl/README.md` for
            the options.
        """
        scout_loc, walls_changed = self.shape_obs(observation)
        if self.curr_turn == 0:
            scout_loc = (0,0)
        
        if scout_loc == (-1,-1):
            for x in range(2 * self.size):
                for y in range(self.size):
                    self.calc_next(self.curr_turn, x, y)
        else:
            self.seen_scout(scout_loc)

        for x in range(self.size):
            for y in range(self.size):
                self.calc_next_repeated(self.curr_turn, x, y)

        self.visualise()

        self.curr_turn += 1
        
        
        # ego_loc = list(observation["location"])
        # ego_dir = observation["direction"]
        # if ego_dir == Direction.LEFT or ego_dir == Direction.RIGHT: ego_loc[0] += 16
        # ego_loc = tuple(ego_loc)

        # astar_half_grid = np.dstack([self.obs_wall_top_space, self.obs_wall_left_space, self.obs_wall_bottom_space, self.obs_wall_right_space])
        # astar_grid = np.tile(astar_half_grid, (2,1,1))
        # astar_path = astar.find_path(astar_grid, ego_loc, scout_loc)
        # try:
        #     next_loc = astar_path[1]
        # except:
        #     next_loc = ego_loc
            
        # # --- LATIN AMERICAN Cheese ---
        # try:
        #     next_next_loc = astar_path[2]
        # except:
        #     next_next_loc = next_loc
            
        # action = randint(0,3)
        # if next_loc[0] == ego_loc[0] + 16 or next_loc[0] == ego_loc[0] - 16: # change dimension
        #     # --- LATIN AMERICAN Cheese ---
        #     if next_next_loc[1] == next_loc[1] - 1: # move up
        #         if ego_dir == Direction.LEFT: action = 3 # turn right
        #         elif ego_dir == Direction.RIGHT: action = 2 # turn left
        #     elif next_next_loc[1] == next_loc[1] + 1: # move down
        #         if ego_dir == Direction.LEFT: action = 2 # turn left
        #         elif ego_dir == Direction.RIGHT: action = 3 # turn right
        #     elif next_next_loc[0] == next_loc[0] - 1: # move left
        #         if ego_dir == Direction.UP: action = 2 # turn left
        #         elif ego_dir == Direction.DOWN: action = 3 # turn right
        #     elif next_next_loc[0] == next_loc[0] + 1: # move right
        #         if ego_dir == Direction.UP: action = 3 # turn right
        #         elif ego_dir == Direction.DOWN: action = 2 # turn left
            
        #     # action = 2 # turn left
        # elif next_loc[1] == ego_loc[1] - 1: # move up
        #     if ego_dir == Direction.UP: action = 0 # move forward
        #     elif ego_dir == Direction.DOWN: action = 1 # move backward
        # elif next_loc[1] == ego_loc[1] + 1: # move down
        #     if ego_dir == Direction.UP: action = 1 # move backward
        #     elif ego_dir == Direction.DOWN: action = 0 # move forward
        # elif next_loc[0] == ego_loc[0] - 1: # move left
        #     if ego_dir == Direction.LEFT: action = 0 # move forward
        #     elif ego_dir == Direction.RIGHT: action = 1 # move backward
        # elif next_loc[0] == ego_loc[0] + 1: # move right
        #     if ego_dir == Direction.LEFT: action = 1 # move backward
        #     elif ego_dir == Direction.RIGHT: action = 0 # move forward

        return 4
            
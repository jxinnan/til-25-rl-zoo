from itertools import cycle
from random import randint, random

import numpy as np
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper

import til_environment.gridworld_astar as astar
from til_environment.types import Action, Direction, Tile

class EvalWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env, running_scout, guard_classes = None, scout_class = None, chosen_astar = (-1,-1), side_astar = (-1,-1)):
        """
        Parameters
        ----------
        running_scout : bool, default True
            `True` if testing/training a scout, `False` if guard
        guard_classes : list of classes (not class instances), optional
            Optional if you are hard-coding other agents in `step()` e.g. for god-mode\n
            Length of list should be 3 if training a scout, 2 if training a guard\n
            Class follows definiton of `RLManager`, so it must have the following method:\n
                `.rl(self, observation: dict[str, int | list[int]]) -> int`
        scout_class : class (not class instance), optional
            See `guard_agents` for class specifications\n
            Should be `None` if training a scout
        chosen_astar : tuple of (float or int)
            Between 0 and 1 (inclusive), it represents the probability the guard follows the A* algorithm instead of sampling a random action.
            If -1, the guard would not move.
            The first number is used when the scout is not in sight.
            The second number is used when the scout is within the viewcone.
        side_astar : tuple of (float or int)
            Between 0 and 1 (inclusive), it represents the probability the other two guards follows the A* algorithm instead of sampling a random action.
            If -1, the guard would not move.
            The first number is used when the scout is not in sight.
            The second number is used when the scout is within the viewcone.
        """
        super().__init__(env)

        self.running_scout = running_scout
        self.guard_classes = guard_classes
        self.scout_class = scout_class if not running_scout else None
        self.chosen_astar = chosen_astar
        self.side_astar = side_astar

        self.obs_wall_top_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_left_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_bottom_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_right_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_rewards_space = np.full((self.size, self.size), 255, dtype=np.uint8)

        self.obs_guard_space = np.full((self.size, self.size), 16, dtype=np.uint8)

        self.true_wall_grid = np.zeros((self.size, self.size, 4), dtype=np.uint8)

        # deadend info - 1 for every step into a deadend
        self.true_deadend_depth_grid = np.zeros((self.size, self.size), dtype=np.uint8)

        self.consecutive_turns = 0
    
    def step(self, the_chosen_action):
        if the_chosen_action is not None:
            super().step(the_chosen_action)

        # custom rewards based on action
        if the_chosen_action in (Action.LEFT, Action.RIGHT,):
            self._cumulative_rewards[self.the_chosen_one] += self.rewards_dict.get(
                "custom_TURN", 0
            )
            self.consecutive_turns += 1
            if self.consecutive_turns >= 3:
                self._cumulative_rewards[self.the_chosen_one] += self.rewards_dict.get(
                    "custom_THREE_TURNS", 0
                )
        elif the_chosen_action != Action.STAY:
            self.consecutive_turns = 0

        for agent in self.agent_iter():
            observation, reward, termination, truncation, info = self.last()

            if agent == self.the_chosen_one:
                break
            elif termination or truncation:
                observation = self.observe(self.the_chosen_one)
                self.shape_obs(observation)
                return (
                    observation, 
                    self._cumulative_rewards[self.the_chosen_one],
                    termination,
                    truncation,
                    self.infos[self.the_chosen_one],
                )
            elif agent == self.scout:
                if self.scout_class != None:
                    action = self.scout_agent.rl(observation)
                else:
                    # TODO: Insert your hard-coded SCOUT policy here
                    action = self.action_space.sample() # anyhow whack
            else:
                if self.guard_classes != None:
                    guard_manager = next(self.guard_agents)
                    action = guard_manager.rl(observation)
                else:
                    # TODO: Insert your hard-coded GUARD policy here
                    action = 4

                    # START A*
                    if agent == self.the_chosen_guard: astar_prob_tuple = self.chosen_astar
                    else: astar_prob_tuple = self.side_astar
                    astar_prob = astar_prob_tuple[0]

                    break_flag = False
                    for y_arr in observation["viewcone"]:
                        if break_flag: break
                        for x_obj in y_arr:
                            if np.unpackbits(x_obj)[5] == 1: # has scout
                                astar_prob = astar_prob_tuple[1]
                                break_flag = True
                                break
                        
                    if astar_prob != -1:
                        ego_loc = list(observation["location"])
                        ego_dir = observation["direction"]
                        if ego_dir == Direction.LEFT or ego_dir == Direction.RIGHT: ego_loc[0] += 16
                        ego_loc = tuple(ego_loc)
                        scout_loc = tuple(self.agent_locations[self.scout])

                        astar_half_grid = self.true_wall_grid
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

                        if next_loc[0] == ego_loc[0] + 16 or next_loc[0] == ego_loc[0] - 16: # change dimension
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
                        # END A*

                        # START 0.5A*
                        if astar_prob < 1:
                            if random() > astar_prob: action = randint(0,4)
                        # END 0.5A*


            super().step(action)
        
        # decay seen guards over 16 steps to 128
        # self.obs_guard_space[self.obs_guard_space > 128] -= np.minimum(self.obs_guard_space[self.obs_guard_space > 128] - 128, 8)
        # self.obs_guard_space[self.obs_guard_space < 128] += np.minimum(128 - self.obs_guard_space[self.obs_guard_space < 128], 8)
        self.obs_guard_space[self.obs_guard_space > 16] -= 1
        self.obs_guard_space[self.obs_guard_space < 16] += 1

        # START custom rewards based on observation
        new_gridview = observation["viewcone"]
        curr_direction = observation["direction"] # right down left up
        curr_location = observation["location"]

        if self.true_deadend_depth_grid[*curr_location] > 0:
            reward += self.rewards_dict.get(
                "custom_DEADEND_BASE", 0
            )

        # rotate clockwise so absolute north faces up
        new_gridview = np.rot90(new_gridview, k=curr_direction)

        match curr_direction: # location of self in rotated new_gridview
            case Direction.RIGHT: rel_curr_location = (2,2)
            case Direction.DOWN: rel_curr_location = (2,2)
            case Direction.LEFT: rel_curr_location = (4,2)
            case Direction.UP: rel_curr_location = (2,4)

        # update tile by tile, column by column, in global POV
        for i in range(new_gridview.shape[0]):
            new_abs_x = curr_location[0] + i - rel_curr_location[0]
            if new_abs_x < 0 or new_abs_x >= self.size: continue
            for j in range(new_gridview.shape[1]):
                new_abs_y = curr_location[1] + j - rel_curr_location[1]
                if new_abs_y < 0 or new_abs_y >= self.size: continue

                # extract data
                unpacked = np.unpackbits(new_gridview[i, j])
                tile_contents = np.packbits(np.concatenate((np.zeros(6, dtype=np.uint8), unpacked[-2:])))[0]
                if tile_contents != Tile.NO_VISION and self.obs_rewards_space[new_abs_x, new_abs_y] == np.uint8(255):
                    reward += self.rewards_dict.get(
                        "custom_SEE_NEW", 0
                    )
        # END custom rewards from observation

        self.shape_obs(observation)
        return observation, reward, termination, truncation, info
    
    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        if self.running_scout:
            self.the_chosen_one = self.scout

            # differentiating guards, selecting one main guard
            for candidate_agent in self.agents:
                if candidate_agent == self.scout: continue
                self.the_chosen_guard = candidate_agent
                break
        else:
            the_chosen_index = randint(0, len(self.agents) - 2)
            for candidate_agent in self.agents:
                if candidate_agent == self.scout: continue
                if the_chosen_index == 0:
                    self.the_chosen_one = candidate_agent
                    break
                the_chosen_index -= 1
            
            self.scout_agent = self.scout_class()
        
        if self.guard_classes:
            self.guard_agents = cycle([klass() for klass in self.guard_classes])
        
        # global information for god guards
        for x, y in np.ndindex((self.size, self.size)):
            self.true_wall_grid[x,y] = np.unpackbits(self.state()[x, y])[:4]
        
        self.true_deadend_depth_grid = np.zeros((self.size, self.size), dtype=np.uint8)
        # deadend depth information
        deadend_final_tiles = []
        for x, y in np.ndindex((self.size, self.size)): # find all deadends
            if np.count_nonzero(self.true_wall_grid[x,y]) >= 3: # deadend destination
                deadend_final_tiles.append((x,y,np.where(self.true_wall_grid[x,y] == 0)[0]))

        deadend_all_tiles = [[] for i in range(len(deadend_final_tiles))]
        for i in range(len(deadend_final_tiles)):
            curr_x = deadend_final_tiles[i][0]
            curr_y = deadend_final_tiles[i][1]
            curr_opening_dir = deadend_final_tiles[i][2][0]

            while True: # alleyway with two openings
                deadend_all_tiles[i].insert(0, (curr_x, curr_y))
                match curr_opening_dir:
                    case 0: # top
                        curr_y -= 1
                        invalid_next_opening_dir = 2
                    case 1: # left
                        curr_x -= 1
                        invalid_next_opening_dir = 3
                    case 2: # bottom
                        curr_y += 1
                        invalid_next_opening_dir = 0
                    case 3: # right
                        curr_x += 1
                        invalid_next_opening_dir = 1
                
                next_opening_arr = np.where(self.true_wall_grid[curr_x,curr_y] == 0)[0]
                if next_opening_arr.shape[0] > 2: # out of deadend
                    break
                
                for opening in next_opening_arr: # find other opening
                    if opening != invalid_next_opening_dir:
                        curr_opening_dir = opening
                        break
        
        for deadend_list in deadend_all_tiles:
            for i in range(len(deadend_list)):
                x = deadend_list[i][0]
                y = deadend_list[i][1]
                self.true_deadend_depth_grid[x, y] = i + 1

        # reset observation shaping
        self.obs_wall_top_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_left_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_bottom_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_right_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_rewards_space = np.full((self.size, self.size), 255, dtype=np.uint8)
        # self.obs_guard_space = np.full((self.size, self.size), 128, dtype=np.uint8)
        self.obs_guard_space = np.full((self.size, self.size), 16, dtype=np.uint8)

        self.consecutive_turns = 0

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        # return self.shape_obs(self.observe(self.the_chosen_one)), self.infos[self.the_chosen_one]
        step_results = self.step(None)
        return step_results[0], step_results[4]

    def shape_obs(self, obs):
        # TODO: Add any observation shaping here
        # e.g. persisting seen tiles
        # output here must match self.observation_space (defined in __init__)
        # this sample simulates FlattenDictWrapper cos it's bwoken :(
        # return flatten(self.observation_space_dict, obs)

        # np_arr[x,y] stores information for coordinate (x,y), where x is horizontal, y is vertical
        # in this case, np.rot90(np_arr) by default rotates it "CLOCKWISE" (for imagination purposes)

        new_gridview = obs["viewcone"]
        curr_direction = obs["direction"] # right down left up
        curr_location = obs["location"]

        # rotate clockwise so absolute north faces up
        new_gridview = np.rot90(new_gridview, k=curr_direction)

        match curr_direction: # location of self in rotated new_gridview
            case Direction.RIGHT: rel_curr_location = (2,2)
            case Direction.DOWN: rel_curr_location = (2,2)
            case Direction.LEFT: rel_curr_location = (4,2)
            case Direction.UP: rel_curr_location = (2,4)

        # update tile by tile, column by column, in global POV
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

                    self.obs_rewards_space[new_abs_x, new_abs_y] = (tile_contents - 1) * np.uint8(5*17)

                # update visible guards
                tile_guard_info = unpacked[4]
                if tile_guard_info == 1:
                    # self.obs_guard_space[new_abs_x, new_abs_y] = np.uint8(255)
                    self.obs_guard_space[new_abs_x, new_abs_y] = np.uint8(32)
                else:
                    self.obs_guard_space[new_abs_x, new_abs_y] = np.uint8(0)

        return obs
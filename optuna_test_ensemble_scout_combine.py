from enum import IntEnum
from importlib import import_module

import numpy as np
import optuna
from tqdm import tqdm

from til_environment import gridworld
from til_environment.types import RewardNames
from til_environment.wrappers import EvalWrapper

import gridworld_astar as astar

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

class EnsembleRLManager:
    def __init__(self, voter_register, weights_peace, weights_war, war_thresh):
        assert len(voter_register) == len(weights_peace) == len(weights_war)

        self.voters = [getattr(import_module(f"ensemble_scouts.{voter_name}.rl_manager"), "RLManager")() for voter_name in voter_register]
        self.weights_peace = weights_peace
        self.weights_war = weights_war
        self.war_thresh = war_thresh

        self.action = 4
        
        self.size = 16
        self.obs_wall_top_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_left_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_bottom_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_wall_right_space = np.zeros((self.size, self.size), dtype=np.uint8)
        self.obs_guard_space = np.full((self.size, self.size), 255, dtype=np.uint8)
    
    def astar_dist(self, location, dst_loc):
        ego_loc = tuple(location)
        dst_loc = tuple(dst_loc)

        astar_half_grid = np.dstack([self.obs_wall_top_space, self.obs_wall_left_space, self.obs_wall_bottom_space, self.obs_wall_right_space])
        astar_grid = np.tile(astar_half_grid, (2,1,1))
        astar_path_1 = astar.find_path(astar_grid, ego_loc, dst_loc)
        astar_path_2 = astar.find_path(astar_grid, (ego_loc[0]+16, ego_loc[1]), dst_loc)
        
        return min(len(astar_path_1), len(astar_path_2))

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        seen_guard = False
        self.obs_guard_space[self.obs_guard_space < 255] += 1

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
        for i in range(new_gridview.shape[0]):
            new_abs_x = curr_location[0] + i - rel_curr_location[0]
            if new_abs_x < 0 or new_abs_x >= self.size: continue
            
            for j in range(new_gridview.shape[1]):
                new_abs_y = curr_location[1] + j - rel_curr_location[1]
                if new_abs_y < 0 or new_abs_y >= self.size: continue

                # extract data
                unpacked = np.unpackbits(new_gridview[i, j])
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
                tile_guard_info = unpacked[4]
                if tile_guard_info == 1:
                    self.obs_guard_space[new_abs_x, new_abs_y] = np.uint8(0)
                    seen_guard = True
                else:
                    self.obs_guard_space[new_abs_x, new_abs_y] = np.uint8(255)

        if seen_guard:
            weights = self.weights_war
        else:
            weights = self.weights_peace
            for dst_loc in np.argwhere(self.obs_guard_space < self.war_thresh):
                if self.astar_dist(curr_location, dst_loc) + self.obs_guard_space[*dst_loc] <= self.war_thresh:
                    weights = self.weights_war
                    break

        ballot_box = [weight * voter.rl(observation, self.action) for voter, weight in zip(self.voters, weights)]
        
        election_results = np.zeros((4,))
        for vote in ballot_box:
            assert len(vote) == 4
            election_results += vote
        
        self.action = np.argmax(election_results)

        return self.action
    
def is_set(x, n):
    """
    Returns if the specified bit (indexed from the smallest power) is set.
    """
    return x & (1 << n) != 0

def get_four_bits(x, n):
    """
    Returns 4 bits, starting from the specified bit (indexed from the smallest power) as the smallest.
    """
    return (x & (0b1111 << n)) >> n

def give_env(custom_rewards_dict = None, custom_render_mode = None, main_guard = (-1,-1), side_guard = (-1,-1)):
    return EvalWrapper(
        gridworld.env(
            env_wrappers = [],
            render_mode = custom_render_mode,
            debug = True,
            novice = False,
            rewards_dict = custom_rewards_dict,
        ), 
        running_scout = True, 
        guard_classes = None,
        scout_class = None,
        chosen_astar = main_guard,
        side_astar = side_guard,
    )

def test_scout(scout_name, scout_manager_kwargs = {}, no_of_matches = 100, test_with_guards = True, save = True):
    SAMPLE_SIZE = no_of_matches
    TEST_SEEDS = [i for i in range(1, SAMPLE_SIZE+1)]

    TEST_GUARDS = [
        # {"main_guard": (1,1), "side_guard": (-1,-1)},
        {"main_guard": (0.375,1), "side_guard": (0.375,1)},
        # {"main_guard": (1,1), "side_guard": (1,1)},
    ] if test_with_guards else []

    # max 1 per step - 1-bit
    NOGUARD_REWARDS = [
        RewardNames.SCOUT_RECON,
        RewardNames.SCOUT_MISSION,
        RewardNames.WALL_COLLISION,
        RewardNames.STATIONARY_PENALTY,
        "custom_THREE_TURNS",
        "custom_DEADEND_BASE",
    ]

    # max 15 per step - 4-bits
    NOGUARD_REWARDS_MULTI = [
        "custom_SEE_NEW",
    ]

    NOGUARD_REWARDS_COMBINED = NOGUARD_REWARDS + NOGUARD_REWARDS_MULTI

    # max 1 per step - 1-bit
    GUARD_REWARDS = [
        RewardNames.SCOUT_RECON,
        RewardNames.SCOUT_MISSION,
        RewardNames.WALL_COLLISION,
        RewardNames.STATIONARY_PENALTY,
        "custom_THREE_TURNS",
        "custom_DEADEND_BASE",
        RewardNames.SCOUT_TRUNCATION,
    ]

    # max 15 per step - 4-bits
    GUARD_REWARDS_MULTI = [
        "custom_SEE_NEW",
    ]

    GUARD_REWARDS_COMBINED = GUARD_REWARDS + GUARD_REWARDS_MULTI

    NOGUARD_REWARDS_DICT = {}
    for i in range(len(NOGUARD_REWARDS)):
        NOGUARD_REWARDS_DICT[NOGUARD_REWARDS[i]] = 2 ** i
    for i in range(len(NOGUARD_REWARDS_MULTI)):
        NOGUARD_REWARDS_DICT[NOGUARD_REWARDS_MULTI[i]] = 2 ** (i * 4 + len(NOGUARD_REWARDS))

    GUARD_REWARDS_DICT = {}
    for i in range(len(GUARD_REWARDS)):
        GUARD_REWARDS_DICT[GUARD_REWARDS[i]] = 2 ** i
    for i in range(len(GUARD_REWARDS_MULTI)):
        GUARD_REWARDS_DICT[GUARD_REWARDS_MULTI[i]] = 2 ** (i * 4 + len(GUARD_REWARDS))
    
    # total_noguard_rewards = np.zeros((len(NOGUARD_REWARDS_DICT), SAMPLE_SIZE,))
    total_guard_rewards = np.zeros((len(GUARD_REWARDS_DICT), SAMPLE_SIZE,))

    # noguard_ep_len = np.zeros((SAMPLE_SIZE,))
    guard_ep_len = np.zeros((SAMPLE_SIZE,))

    for seed_idx in tqdm(range(len(TEST_SEEDS))):
        seed = TEST_SEEDS[seed_idx]
        # model = EnsembleRLManager(**scout_manager_kwargs)
        # round_rewards = np.zeros((len(NOGUARD_REWARDS_DICT),))

        # env = give_env(NOGUARD_REWARDS_DICT)
        # obs, _info = env.reset(seed=seed)

        # ep_len = 0
        # terminated = False
        # truncated = False
        # while not terminated and not truncated:
        #     action = model.rl(obs)
        #     obs, reward, terminated, truncated, _info = env.step(action)
        #     ep_len += 1

        #     for rew_idx in range(len(NOGUARD_REWARDS)):
        #         if is_set(reward, rew_idx):
        #             round_rewards[rew_idx] += 1

        #     for multi_rew_idx in range(len(NOGUARD_REWARDS_MULTI)):
        #         rew_idx = multi_rew_idx * 4 + len(NOGUARD_REWARDS)
        #         round_rewards[rew_idx] += get_four_bits(reward, rew_idx)
        
        # total_noguard_rewards[:, seed_idx] = round_rewards
        # noguard_ep_len[seed_idx] = ep_len
        # env.close()

        guard_idx = 0
        guard_kwargs = TEST_GUARDS[guard_idx]
        model = EnsembleRLManager(**scout_manager_kwargs)
        round_rewards = np.zeros((len(GUARD_REWARDS_DICT),))

        env = give_env(GUARD_REWARDS_DICT, **guard_kwargs)
        obs, _info = env.reset(seed=seed)

        ep_len = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = model.rl(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_len += 1

            for rew_idx in range(len(GUARD_REWARDS)):
                if is_set(reward, rew_idx):
                    round_rewards[rew_idx] += 1

            for multi_rew_idx in range(len(GUARD_REWARDS_MULTI)):
                rew_idx = multi_rew_idx * 4 + len(GUARD_REWARDS)
                round_rewards[multi_rew_idx + len(GUARD_REWARDS)] += get_four_bits(reward, rew_idx)
        
        total_guard_rewards[:, seed_idx] = round_rewards
        guard_ep_len[seed_idx] = ep_len
        env.close()

    return np.mean(total_guard_rewards[0]) + np.mean(total_guard_rewards[1]) * 4, len(guard_ep_len[guard_ep_len < 100]) / len(guard_ep_len) * 100
    
    # return len(guard_ep_len[guard_ep_len < 100]) / len(guard_ep_len)

VOTERS = [
    "ensemble_fellowship_v2_full",
    "ensemble_runhidetell_v2",
]

def objective(trial):
    peace_gandhi = trial.suggest_float("peace_gandhi", 0, 1)
    war_gandhi = trial.suggest_float("war_gandhi", 0, 1)
    war_thresh = trial.suggest_int("war_thresh", 2, 20)

    # p = []
    # for i in range(n):
    #     p.append(x[i] / sum(x))

    # for i in range(n):
    #     trial.set_user_attr(f"p_{i}", p[i])
    
    mean_score, death_score = test_scout(
        scout_name="trial",
        scout_manager_kwargs={
            "voter_register": VOTERS,
            "weights_peace": [peace_gandhi, 1-peace_gandhi],
            "weights_war": [war_gandhi, 1-war_gandhi],
            "war_thresh": war_thresh,
        },
        no_of_matches=100,
        test_with_guards=True,
        save=False,
    )

    trial.set_user_attr("mean_score", mean_score)
    trial.set_user_attr("death_score", death_score)

    return 200 - (mean_score - death_score)

def main():
    # storage = optuna.storages.JournalStorage(
    #     optuna.storages.journal.JournalFileBackend("logs/optuna/fellowship_optuna.log"),
    # )
    # study = optuna.create_study(
    #     sampler=optuna.samplers.TPESampler(),
    #     storage=storage,
    # )
    study = optuna.load_study(
        study_name="gandhi",
        storage="mysql+pymysql://root@localhost/gandhi",
    )
    study.optimize(objective, n_trials=80)

    # print(study.best_params)
          
    # n = 12
    # p = []
    # for i in range(n):
    #     p.append([trial.user_attrs[f"p_{i}"] for trial in study.trials])
    # axes = plt.subplots(n, n, figsize=(20, 20))[1]

    # for i in range(n):
    #     for j in range(n):
    #         axes[j][i].scatter(p[i], p[j], marker=".")
    #         axes[j][i].set_xlim(0, 1)
    #         axes[j][i].set_ylim(0, 1)
    #         axes[j][i].set_xlabel(f"p_{i}")
    #         axes[j][i].set_ylabel(f"p_{j}")

    # plt.savefig("sampled_ps.png")

if __name__ == "__main__":
    main()
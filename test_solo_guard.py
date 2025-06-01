import argparse
from importlib import import_module
from multiprocessing import Pool
import os

import numpy as np

from til_environment import gridworld
from til_environment.types import RewardNames
from til_environment.wrappers import GuardEvalWrapper

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

def get_eight_bits(x, n):
    """
    Returns 8 bits, starting from the specified bit (indexed from the smallest power) as the smallest.
    """
    return (x & (0b11111111 << n)) >> n

def give_env(custom_rewards_dict = None, custom_render_mode = None, guard_classes = None, scout_class = None, main_guard = (-1,-1), side_guard = (-1,-1)):
    return GuardEvalWrapper(
        gridworld.env(
            env_wrappers = [],
            render_mode = custom_render_mode,
            debug = True,
            novice = False,
            rewards_dict = custom_rewards_dict,
        ), 
        running_scout = False, 
        guard_classes = guard_classes,
        scout_class = scout_class,
        chosen_astar = main_guard,
        side_astar = side_guard,
    )

parser = argparse.ArgumentParser()
parser.add_argument("guard_name")
parser.add_argument("-n", "--no_of_matches", type=int, default=200)
parser.add_argument("--hybrid", action='store_true')
parser.add_argument("-s", "--save", action='store_true')
parser.add_argument("--nproc", type=int, default=os.cpu_count())
parser.add_argument("--ipynb", action='store_true')
args = parser.parse_args()

if args.ipynb:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

if args.hybrid:
    RLManager = getattr(import_module(f"hybrid.{args.guard_name}.rl_manager"), "RLManager")
else:
    RLManager = getattr(import_module(f"guards.{args.guard_name}.rl_manager"), "RLManager")

SAMPLE_SIZE = args.no_of_matches
TEST_SEEDS = [i for i in range(1, SAMPLE_SIZE+1)]

# scout
stationary_scout = getattr(import_module(f"hybrid.stationary.rl_manager"), "RLManager")
wonder = getattr(import_module(f"scouts.wonder.rl_manager"), "RLManager")
avignon = getattr(import_module(f"scouts.avignon-prep-8M.rl_manager"), "RLManager")
atlanta = getattr(import_module(f"scouts.atlanta-8M-guard-twowalls-8M.rl_manager"), "RLManager")

# hybrid
mcts = getattr(import_module(f"hybrid.mcts_depth7.rl_manager"), "RLManager")
cnnppo = getattr(import_module(f"hybrid.cnnppo_split_v1.rl_manager"), "RLManager")

# guard
helvetica = getattr(import_module(f"guards.helvetica.rl_manager"), "RLManager")

TEST_SCOUTS = [
    # {"scout_class": stationary_scout},
    {"scout_class": wonder},
    {"scout_class": avignon},
    {"scout_class": mcts},
    {"scout_class": cnnppo},
    {"scout_class": atlanta},
]

TEST_GUARDS = [
    {"main_guard": (-1,-1), "side_guard": (-1,-1)},
    {"main_guard": (0.375,1), "side_guard": (0.375,1)},
    {"guard_classes":[helvetica, mcts]},
    {"guard_classes":[mcts, cnnppo]},
]

# max 1 per step - 1-bit
REWARDS = [
    "custom_SEE_SCOUT",
    "custom_SCOUT_ONE_TILE",
    RewardNames.WALL_COLLISION,
    RewardNames.STATIONARY_PENALTY,
    "custom_THREE_TURNS",
    RewardNames.GUARD_CAPTURES,
    RewardNames.GUARD_WINS,
    RewardNames.AGENT_COLLIDER,
]

# max 15 per step - 4-bits
REWARDS_MULTI_QUAD = [
    RewardNames.AGENT_COLLIDEE,
    "custom_SEE_NEW",
]

# max 255 per step - 8-bits
REWARDS_MULTI_OCTO = [
    "custom_DIST_TO_SCOUT",
]

REWARDS_COMBINED = REWARDS + REWARDS_MULTI_QUAD + REWARDS_MULTI_OCTO

REWARDS_DICT = {}
for i in range(len(REWARDS)):
    REWARDS_DICT[REWARDS[i]] = 2 ** i
for i in range(len(REWARDS_MULTI_QUAD)):
    REWARDS_DICT[REWARDS_MULTI_QUAD[i]] = 2 ** (i * 4 + len(REWARDS))
for i in range(len(REWARDS_MULTI_OCTO)):
    REWARDS_DICT[REWARDS_MULTI_OCTO[i]] = 2 ** (i * 8 + len(REWARDS_MULTI_QUAD) * 4 + len(REWARDS))

total_rewards = np.zeros((len(TEST_SCOUTS), len(TEST_GUARDS), len(REWARDS_DICT), SAMPLE_SIZE,))

total_ep_len = np.zeros((len(TEST_SCOUTS), len(TEST_GUARDS), SAMPLE_SIZE,))

def run_scout_guard_pair(input_args):
    scout_kwargs = input_args[0]
    guard_kwargs = input_args[1]
    seed = input_args[2]

    round_rewards = np.zeros((len(REWARDS_DICT),))
    round_ep_len = 0

    chosen_pos_index = seed%3

    model = RLManager()

    env = give_env(REWARDS_DICT, **scout_kwargs, **guard_kwargs)
    obs, _info = env.reset(seed=seed, chosen_guard_index=chosen_pos_index)

    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = model.rl(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        round_ep_len += 1

        for rew_idx in range(len(REWARDS)):
            if is_set(reward, rew_idx):
                round_rewards[rew_idx] += 1

        for multi_rew_idx in range(len(REWARDS_MULTI_QUAD)):
            rew_idx = multi_rew_idx * 4 + len(REWARDS)
            round_rewards[multi_rew_idx + len(REWARDS)] += get_four_bits(reward, rew_idx)
        
        for multi_rew_idx in range(len(REWARDS_MULTI_OCTO)):
            rew_idx = multi_rew_idx * 8 + len(REWARDS_MULTI_QUAD) * 4 + len(REWARDS)
            round_rewards[multi_rew_idx + len(REWARDS_MULTI_QUAD) + len(REWARDS)] += get_eight_bits(reward, rew_idx)
    
    env.close()

    round_rewards[-1] /= round_ep_len # just for DIST_TO_SCOUT

    return round_rewards, round_ep_len

for seed_idx in tqdm(range(len(TEST_SEEDS))):
    seed = TEST_SEEDS[seed_idx]
    
    mp_args = []
    for scout_idx in range(len(TEST_SCOUTS)):
        scout_kwargs = TEST_SCOUTS[scout_idx]
        for guard_idx in range(len(TEST_GUARDS)):
            guard_kwargs = TEST_GUARDS[guard_idx]
            mp_args.append([
                scout_kwargs,
                guard_kwargs,
                seed,
            ])
    
    with Pool(args.nproc) as p:
        mp_out = p.map(run_scout_guard_pair, mp_args)
    
    mp_idx = 0
    for scout_idx in range(len(TEST_SCOUTS)):
        for guard_idx in range(len(TEST_GUARDS)):
            round_rewards = mp_out[mp_idx][0]
            round_ep_len = mp_out[mp_idx][1]
            mp_idx += 1

            total_rewards[scout_idx, guard_idx, :, seed_idx] = round_rewards
            total_ep_len[scout_idx, guard_idx, seed_idx] = round_ep_len

print("Aggregated")
print(",".join(
    [
            ",".join([
                    rew_name + metric_name
                for metric_name in ["_mean", "_var", "_min", "_max"]
            ])
        for rew_name in REWARDS_COMBINED
    ]
    +[
        ",".join([
                "ep_len" + metric_name 
            for metric_name in ["_mean", "_var", "_min", "_max"]
        ])
    ]
))
print(",".join(
    [
            ",".join([
                str(np.mean(total_rewards[:,:,i,:])),
                str(np.var(total_rewards[:,:,i,:])),
                str(np.min(total_rewards[:,:,i,:])),
                str(np.max(total_rewards[:,:,i,:])),
            ])
        for i in range(len(REWARDS_COMBINED))
    ]
    +[
        ",".join([
            str(np.mean(total_ep_len)),
            str(np.var(total_ep_len)),
            str(np.min(total_ep_len)),
            str(np.max(total_ep_len)),
        ])
    ]
))
print("\nSummary")
rewards_list = [0, 5, 6, 10]
print(",".join([
        ",".join([
                str(np.mean(total_rewards[scout_idx][:,rew,:]))
            for rew in rewards_list
        ]
        +[
            str(np.mean(total_ep_len[scout_idx]))
        ]
        )
    for scout_idx in range(total_rewards.shape[0])
]))

if args.save:
    save_path = f"logs/test_solo_guard/{'hybrid_' if args.hybrid else ''}{args.guard_name}.npz"
    np.savez(
        save_path,
        total_rewards=total_rewards,
        total_ep_len=total_ep_len,
    )
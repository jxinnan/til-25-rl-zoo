import argparse
from importlib import import_module

import numpy as np
from tqdm import tqdm

from til_environment import gridworld
from til_environment.types import RewardNames
from til_environment.wrappers import EvalWrapper

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scout_name")
    parser.add_argument("-n", "--no_of_matches", type=int, default=1000)
    parser.add_argument("-g", "--test_with_guards", action='store_true')
    parser.add_argument("--hybrid", action='store_true')
    parser.add_argument("-s", "--save", action='store_true')
    args = parser.parse_args()

    if args.hybrid:
        RLManager = getattr(import_module(f"hybrid.{args.scout_name}.rl_manager"), "RLManager")
    else:
        RLManager = getattr(import_module(f"scouts.{args.scout_name}.rl_manager"), "RLManager")

    SAMPLE_SIZE = args.no_of_matches
    TEST_SEEDS = [i for i in range(1, SAMPLE_SIZE+1)]

    TEST_GUARDS = [
        {"main_guard": (1,1), "side_guard": (-1,-1)},
        {"main_guard": (0.375,1), "side_guard": (0.375,1)},
        {"main_guard": (1,1), "side_guard": (1,1)},
    ] if args.test_with_guards else []

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
    
    total_noguard_rewards = np.zeros((len(NOGUARD_REWARDS_DICT), SAMPLE_SIZE,))
    total_guard_rewards = np.zeros((len(TEST_GUARDS), len(GUARD_REWARDS_DICT), SAMPLE_SIZE,))

    noguard_ep_len = np.zeros((SAMPLE_SIZE,))
    guard_ep_len = np.zeros((len(TEST_GUARDS), SAMPLE_SIZE,))

    for seed_idx in tqdm(range(len(TEST_SEEDS))):
        seed = TEST_SEEDS[seed_idx]
        model = RLManager()
        round_rewards = np.zeros((len(NOGUARD_REWARDS_DICT),))

        env = give_env(NOGUARD_REWARDS_DICT)
        obs, _info = env.reset(seed=seed)

        ep_len = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = model.rl(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_len += 1

            for rew_idx in range(len(NOGUARD_REWARDS)):
                if is_set(reward, rew_idx):
                    round_rewards[rew_idx] += 1

            for multi_rew_idx in range(len(NOGUARD_REWARDS_MULTI)):
                rew_idx = multi_rew_idx * 4 + len(NOGUARD_REWARDS)
                round_rewards[rew_idx] += get_four_bits(reward, rew_idx)
        
        total_noguard_rewards[:, seed_idx] = round_rewards
        noguard_ep_len[seed_idx] = ep_len
        env.close()

        for guard_idx in range(len(TEST_GUARDS)):
            guard_kwargs = TEST_GUARDS[guard_idx]
            model = RLManager()
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
                    round_rewards[rew_idx] += get_four_bits(reward, rew_idx)
            
            total_guard_rewards[guard_idx, :, seed_idx] = round_rewards
            guard_ep_len[guard_idx, seed_idx] = ep_len
            env.close()

    print("\nNo Guards")

    print(",".join(
        [
                ",".join([
                        rew_name + metric_name
                    for metric_name in ["_mean", "_var", "_min", "_max"]
                ])
            for rew_name in NOGUARD_REWARDS_COMBINED
        ]
        +[
            ",".join([
                    "ep_len" + metric_name 
                for metric_name in ["_mean", "_var", "_min", "_max"]
            ])
        ]
    ))

    print("\nNo Guards")
    print(",".join(
        [
                ",".join([
                    str(np.mean(rew_sum)),
                    str(np.var(rew_sum)),
                    str(np.min(rew_sum)),
                    str(np.max(rew_sum)),
                ])
            for rew_sum in total_noguard_rewards
        ]
        +[
            ",".join([
                str(np.mean(noguard_ep_len)),
                str(np.var(noguard_ep_len)),
                str(np.min(noguard_ep_len)),
                str(np.max(noguard_ep_len)),
            ])
        ]
    ))

    print("\nGuards")
    print(",".join(
        [
                ",".join([
                        rew_name + metric_name
                    for metric_name in ["_mean", "_var", "_min", "_max"]
                ])
            for rew_name in GUARD_REWARDS_COMBINED
        ]
        +[
            ",".join([
                    "ep_len" + metric_name 
                for metric_name in ["_mean", "_var", "_min", "_max"]
            ])
        ]
    ))
    for guard_idx in range(len(TEST_GUARDS)):
        print(f"\nGuard {guard_idx}")

        print(",".join(
            [
                    ",".join([
                        str(np.mean(rew_sum)),
                        str(np.var(rew_sum)),
                        str(np.min(rew_sum)),
                        str(np.max(rew_sum)),
                    ])
                for rew_sum in total_guard_rewards[guard_idx]
            ]
            +[
                ",".join([
                    str(np.mean(guard_ep_len[guard_idx])),
                    str(np.var(guard_ep_len[guard_idx])),
                    str(np.min(guard_ep_len[guard_idx])),
                    str(np.max(guard_ep_len[guard_idx])),
                ])
            ]
        ))
    
    if args.save:
        save_path = f"logs/test_solo_scout/{'hybrid_' if args.hybrid else ''}{args.scout_name}.npz"
        np.savez(save_path, total_guard_rewards=total_guard_rewards, total_noguard_rewards=total_noguard_rewards)

if __name__ == "__main__":
    main()
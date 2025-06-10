import argparse
from importlib import import_module
from itertools import combinations

import numpy as np
from tqdm import tqdm

from til_environment import gridworld
from til_environment.types import RewardNames
from til_environment.wrappers import EvalWrapper

class EnsembleRLManager:
    def __init__(self, voter_register, weights):
        assert len(voter_register) == len(weights)

        self.voters = [getattr(import_module(f"ensemble_scouts.{voter_name}.rl_manager"), "RLManager")() for voter_name in voter_register]
        self.weights = weights

        self.action = 4

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        ballot_box = [weight * voter.rl(observation, self.action) for voter, weight in zip(self.voters, self.weights)]
        
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
        {"main_guard": (1,1), "side_guard": (-1,-1)},
        {"main_guard": (0.375,1), "side_guard": (0.375,1)},
        {"main_guard": (1,1), "side_guard": (1,1)},
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
    
    total_noguard_rewards = np.zeros((len(NOGUARD_REWARDS_DICT), SAMPLE_SIZE,))
    total_guard_rewards = np.zeros((len(TEST_GUARDS), len(GUARD_REWARDS_DICT), SAMPLE_SIZE,))

    noguard_ep_len = np.zeros((SAMPLE_SIZE,))
    guard_ep_len = np.zeros((len(TEST_GUARDS), SAMPLE_SIZE,))

    for seed_idx in tqdm(range(len(TEST_SEEDS))):
        seed = TEST_SEEDS[seed_idx]
        model = EnsembleRLManager(**scout_manager_kwargs)
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
            
            total_guard_rewards[guard_idx, :, seed_idx] = round_rewards
            guard_ep_len[guard_idx, seed_idx] = ep_len
            env.close()

    # print("\nNo Guards")

    # print(",".join(
    #     [
    #             ",".join([
    #                     rew_name + metric_name
    #                 for metric_name in ["_mean", "_var", "_min", "_max"]
    #             ])
    #         for rew_name in NOGUARD_REWARDS_COMBINED
    #     ]
    #     +[
    #         ",".join([
    #                 "ep_len" + metric_name 
    #             for metric_name in ["_mean", "_var", "_min", "_max"]
    #         ])
    #     ]
    # ))

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

    no_guards_str = ",".join(
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
    )

    # print("\nGuards")
    # print(",".join(
    #     [
    #             ",".join([
    #                     rew_name + metric_name
    #                 for metric_name in ["_mean", "_var", "_min", "_max"]
    #             ])
    #         for rew_name in GUARD_REWARDS_COMBINED
    #     ]
    #     +[
    #         ",".join([
    #                 "ep_len" + metric_name 
    #             for metric_name in ["_mean", "_var", "_min", "_max"]
    #         ])
    #     ]
    # ))
    guards_strs = []
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

        guards_strs.append(",".join(
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
    
    if save:
        save_path = f"logs/test_solo_scout/{scout_name}.npz"
        np.savez(
            save_path,
            total_guard_rewards=total_guard_rewards,
            total_noguard_rewards=total_noguard_rewards,
            noguard_ep_len=noguard_ep_len,
            guard_ep_len=guard_ep_len,
        )
    
    return no_guards_str, guards_strs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("proc_id", type=int)
    args = parser.parse_args()

    VOTERS = [
        "atlanta-8M-deadend-8M",
        "avignon-8M",
        "avignon-ariane4-8M",
        "avignon-ariane20-8M",
        "avignon-ariane24-4M",
        "avignon-ariane24-normal-2M4",
        "avignon-ariane25-4M",
        "avignon-ariane25-normal-2M8",
        "caracas-8M",
        "caracas-ariane25-4M",
        "chitose-6M",
        "chitose-8M",
    ]

    CONSTITUENCIES = [constituency for constituency in combinations(VOTERS, 3)]

    SELECTED_ZONES = CONSTITUENCIES[args.proc_id * len(CONSTITUENCIES) // 20 : (args.proc_id + 1) * len(CONSTITUENCIES) // 20]

    WEIGHTS = [
        [0.2, 0.4, 0.4],
        [0.4, 0.2, 0.4],
        [0.4, 0.4, 0.2],
        [0.2, 0.2, 0.6],
        [0.2, 0.6, 0.2],
        [0.6, 0.2, 0.2],
    ]

    CODENAME = "fellowship"
    curr_idx = args.proc_id * len(CONSTITUENCIES) // 20 * len(WEIGHTS)

    no_guard_str_list = []
    guards_strs_list = [[] for i in range(3)]
    logs = []
    for zone in SELECTED_ZONES:
        for weight in WEIGHTS:
            scout_name = f"ensemble_${CODENAME}_${str(curr_idx).zfill(4)}"
            manager_kwargs = {
                "voter_register": zone,
                "weights": weight,
            }
            logs.append(scout_name + str(zone) + str(weight))
            no_guards_str, guards_strs = test_scout(scout_name, manager_kwargs)
            no_guard_str_list.append(no_guards_str)
            for i in range(len(guards_strs_list)):
                guards_strs_list[i].append(guards_strs[i])
            curr_idx += 1
    
    print(args.proc_id * len(CONSTITUENCIES) // 20, (args.proc_id + 1) * len(CONSTITUENCIES) // 20 - 1)
    
    for log in logs:
        print(log)
    
    print("\nNo Guards")
    for stroll in no_guard_str_list:
        print(stroll)
    
    for i in range(len(guards_strs_list)):
        print("\nGuard",i)
        for stroll in guards_strs_list[i]:
            print(stroll)

if __name__ == "__main__":
    main()
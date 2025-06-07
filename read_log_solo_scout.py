import argparse

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scout_name")
    parser.add_argument("--hybrid", action='store_true')
    args = parser.parse_args()

    save_path = f"logs/test_solo_scout/{'hybrid_' if args.hybrid else ''}{args.scout_name}.npz"
    log_obj = np.load(save_path)
    total_guard_rewards = log_obj['total_guard_rewards']
    total_noguard_rewards = log_obj['total_noguard_rewards']
    try:
        noguard_ep_len = log_obj['noguard_ep_len']
        guard_ep_len = log_obj['guard_ep_len']
    except:
        print("No episode length data")

    '''
    NOGUARD_REWARDS = [
        0 RewardNames.SCOUT_RECON,
        1 RewardNames.SCOUT_MISSION,
        2 RewardNames.WALL_COLLISION,
        3 RewardNames.STATIONARY_PENALTY,
        4 "custom_THREE_TURNS",
        5 "custom_DEADEND_BASE",
    ]
    NOGUARD_REWARDS_MULTI = [
        6 "custom_SEE_NEW",
    ]
    '''

    # print(total_noguard_rewards.shape)
    # print(np.argmax(total_noguard_rewards[2]))

    # print(
    #     np.min(total_noguard_rewards[0][np.argwhere(noguard_ep_len == 100)])
    # )

    # print()
    # print(np.argwhere(total_noguard_rewards[0] < 15))

    target_metric = total_noguard_rewards[0]
    comparison = target_metric < 50
    interesting_seeds = []
    for tup in zip(np.argwhere(comparison), target_metric[np.argwhere(comparison)]):
        interesting_seeds.append((tup[0], tup[1], tup[1]+total_noguard_rewards[1][tup[0]], noguard_ep_len[tup[0]]))
        # print(tup[0], tup[1])
    
    for seed in sorted(interesting_seeds, key = lambda x : x[2]):
        if seed[2] < 50 and seed[3] == 100:
            print(seed)

    """
    GUARD_REWARDS = [
        0 RewardNames.SCOUT_RECON,
        1 RewardNames.SCOUT_MISSION,
        2 RewardNames.WALL_COLLISION,
        3 RewardNames.STATIONARY_PENALTY,
        4 "custom_THREE_TURNS",
        5 "custom_DEADEND_BASE",
        6 RewardNames.SCOUT_TRUNCATION,
    ]
    GUARD_REWARDS_MULTI = [
        7 "custom_SEE_NEW",
    ]"""

    target_metric = total_guard_rewards[0][6]
    comparison = target_metric < 1
    interesting_seeds = []
    for tup in zip(np.argwhere(comparison), target_metric[np.argwhere(comparison)]):
        interesting_seeds.append((tup[0], guard_ep_len[0][tup[0]]))
        # interesting_seeds.append(tup[0])
        # print(tup[0], tup[1])
    
    # for seed in sorted(interesting_seeds, key = lambda x : x[1], reverse=True):
    #     if seed[1] < 10:
    #         print(seed)

if __name__ == "__main__":
    main()
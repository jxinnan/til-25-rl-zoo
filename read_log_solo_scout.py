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
    noguard_ep_len = log_obj['noguard_ep_len']
    guard_ep_len = log_obj['guard_ep_len']

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
    target_metric = noguard_ep_len
    comparison = target_metric < 100
    for tup in zip(np.argwhere(comparison), target_metric[np.argwhere(comparison)]):
        print(tup[0], tup[1])

if __name__ == "__main__":
    main()
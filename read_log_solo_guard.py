import argparse

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("guard_name")
    parser.add_argument("--hybrid", action='store_true')
    args = parser.parse_args()

    save_path = f"logs/test_solo_guard/{'hybrid_' if args.hybrid else ''}{args.guard_name}.npz"
    log_obj = np.load(save_path)
    total_rewards = log_obj['total_rewards']
    total_ep_len = log_obj['total_ep_len']

    '''
    TEST_SCOUTS = [
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

    REWARDS = [
        0 "custom_SEE_SCOUT",
        1 "custom_SCOUT_ONE_TILE",        
        2 RewardNames.WALL_COLLISION,
        3 RewardNames.STATIONARY_PENALTY,
        4 "custom_THREE_TURNS",
        5 RewardNames.AGENT_COLLIDER,
        6 RewardNames.AGENT_COLLIDEE,
        7 RewardNames.GUARD_CAPTURES,
        8 RewardNames.GUARD_WINS,
    ]
    REWARDS_MULTI_QUAD = [
        9 "custom_SEE_NEW",
    ]
    REWARDS_MULTI_OCTO = [
        10 "custom_DIST_TO_SCOUT",
    ]
    '''

    print(np.argwhere(total_rewards[:,:,7,:] == 3))

    print(total_rewards.shape)
    print(total_ep_len.shape)

    # rewards_list = [0, 7, 8, 10]
    # print(",".join([
    #         ",".join([
    #                 str(np.mean(total_rewards[scout_idx][:,rew,:]))
    #             for rew in rewards_list
    #         ]
    #         +[
    #             str(np.mean(total_ep_len[scout_idx]))
    #         ]
    #         )
    #     for scout_idx in range(total_rewards.shape[0])
    # ]))

    # print(total_noguard_rewards.shape)
    # print(np.argmax(total_noguard_rewards[2]))
    # target_metric = total_noguard_rewards[0]
    # comparison = target_metric < 24
    # for tup in zip(np.argwhere(comparison), target_metric[np.argwhere(comparison)]):
    #     print(tup[0], tup[1])

if __name__ == "__main__":
    main()
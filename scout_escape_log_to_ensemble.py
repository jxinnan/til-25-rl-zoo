import os

import numpy as np

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

def main():
    MISSION_VALUE = 5
    MIN_THRESH = 0.57

    log_dir = "logs/test_solo_scout/"
    file_paths = [file_path for file_path in os.listdir("logs/test_solo_scout/") 
                  if not (file_path.startswith("mcts") or file_path.startswith("hybrid") or file_path.startswith("wonder") or file_path.startswith("ensemble") or file_path.startswith("odyssey") or file_path.endswith("lessturns.npz") or file_path.endswith("backAndForth.npz"))]

    file_paths = [file_path for file_path in file_paths if 'guard_ep_len' in np.load(os.path.join(log_dir, file_path), allow_pickle=True).keys()]
    model_count = len(file_paths)
    scores = np.zeros((3, 1000, model_count))

    for i in range(model_count):
        file_path = os.path.join(log_dir, file_paths[i])
        log_obj = np.load(file_path, allow_pickle=True)
        guard_ep_len = log_obj['guard_ep_len']
        for k in range(3):
            for j in range(1000):
                scores[k, j, i] = guard_ep_len[k][j]
    # print(model_count)

    ep_lens = np.zeros((3, model_count))
    for i in range(3):
        for j in range(model_count):
            ep_lens[i, j] = len(scores[i,:,j][scores[i,:,j] == 100])
    
    # best_scores = np.max(scores, axis=1)
    # best_scores = np.stack([best_scores for i in range(model_count)], axis=-1)
    # norm_scores = scores/best_scores
    # print(norm_scores.shape)

    # mean_norm = np.mean(norm_scores, axis=0)
    # median_norm = np.median(norm_scores, axis=0)
    # min_norm = np.min(norm_scores, axis=0)
    # print(model_count)
    # print(np.argsort(mean_norm[mean_norm > 0.82]).shape)
    # print(np.argsort(median_norm[median_norm > 0.86]).shape)
    # print(np.argsort(min_norm[min_norm > 0.35]).shape)
    candidates = np.unique(np.concat((np.argsort(ep_lens[0])[-2:],np.argsort(ep_lens[1])[-2:],np.argsort(ep_lens[2])[-2:])))
    for candidate in candidates:
        print(file_paths[candidate][:-4])
    for candidate in candidates:
        print(file_paths[candidate], ep_lens[0, candidate], ep_lens[1, candidate], ep_lens[2, candidate], )
    print(candidates.shape)
    # print(np.unique(candidates).shape)
    # print(np.argmax(mean_norm), np.max(mean_norm))
    # print(np.argmax(min_norm), np.max(min_norm))
    # print(file_paths[np.argmax(mean_norm)])
    # print(file_paths[np.argmax(min_norm)])

    # var_norm = np.var(norm_scores, axis=0)
    # print(np.argmin(var_norm), np.min(var_norm))
    # print(file_paths[np.argmin(var_norm)])

    # seeds = [i for i in range(1000)]
    # models_chosen = []
    # models_weight = []
    # iterations = 0
    # while len(seeds) > 0:
    #     # print(iterations)
    #     iterations += 1
    #     qual_count = np.zeros((model_count))
    #     model_seeds = [[] for i in range(model_count)]
    #     for model_idx in range(model_count):
    #         for seed in seeds:
    #             if norm_scores[seed, model_idx] >= MIN_THRESH:
    #                 qual_count[model_idx] += 1
    #                 model_seeds[model_idx].append(seed)
    #     best_model_idx = np.argmax(qual_count)
    #     models_chosen.append(best_model_idx)
    #     models_weight.append(qual_count[best_model_idx])
    #     for seed in model_seeds[best_model_idx]:
    #         seeds.remove(seed)
    #         # print(len(seeds))
    # for i in range(len(models_chosen)):
    #     print(models_chosen[i], file_paths[models_chosen[i]])
    #     print(models_weight[i])
    #     print(np.mean(norm_scores[:,models_chosen[i]]))
    #     print(np.min(norm_scores[:,models_chosen[i]]))
    #     print(np.var(norm_scores[:,models_chosen[i]]))
    #     print()
    # print(len(models_chosen))
    # qualified = []
    # best_scores = np.zeros((1000))
    # for seed in range(1000):
    #     max_score = np.max(scores[seed])
    #     best_scores[seed] = max_score
    #     qualified.append(np.argwhere(scores[seed] >= MIN_THRESH * max_score))
    
    # must_haves = [int(qual_idx[0][0]) for qual_idx in qualified if len(qual_idx) == 1]
    # for seed in range(1000):
    #     for id in qualified[seed]:
    #         if id in must_haves:
    #             break
    # print(len(set(must_haves)))
    # print(qualified[0])
    # print(np.min(best_scores))

    # the_goods = np.argwhere(scores == np.max(scores, axis=1))
    # print(the_goods)
    # # print(len(set(the_goods)))
    # print(model_count)
    # print(len(the_goods))


    # try:
    #     noguard_ep_len = log_obj['noguard_ep_len']
    #     guard_ep_len = log_obj['guard_ep_len']
    # except:
    #     print("No episode length data")

    # print(total_noguard_rewards.shape)
    # print(np.argmax(total_noguard_rewards[2]))

    # print(
    #     np.min(total_noguard_rewards[0][np.argwhere(noguard_ep_len == 100)]),
    #     np.min(total_noguard_rewards[1][np.argwhere(noguard_ep_len == 100)])
    # )

    # print()
    # print(np.argwhere(total_noguard_rewards[0] < 15))

    # target_metric = total_noguard_rewards[0]
    # comparison = target_metric < 50
    # interesting_seeds = []
    # for tup in zip(np.argwhere(comparison), target_metric[np.argwhere(comparison)]):
    #     interesting_seeds.append((tup[0], tup[1], tup[1]+total_noguard_rewards[1][tup[0]], noguard_ep_len[tup[0]]))
    #     # print(tup[0], tup[1])
    
    # for seed in sorted(interesting_seeds, key = lambda x : x[2]):
    #     if seed[2] < 50:
    #         print(seed)


    # target_metric = total_guard_rewards[0][6]
    # comparison = target_metric == 1
    # interesting_seeds = []
    # for tup in zip(np.argwhere(comparison), target_metric[np.argwhere(comparison)]):
    #     interesting_seeds.append((tup[0], guard_ep_len[0][tup[0]], total_guard_rewards[0][0][tup[0]]+total_guard_rewards[0][1][tup[0]]))
    #     # interesting_seeds.append(tup[0])
    #     # print(tup[0], tup[1])
    
    # for seed in sorted(interesting_seeds, key = lambda x : x[2], reverse=False):
    #     if seed[2] > 20:
    #         print(seed)

if __name__ == "__main__":
    main()
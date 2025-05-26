import argparse
from importlib import import_module

import numpy as np
from tqdm import tqdm

from til_environment import gridworld

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s1", "--scout_1")
    parser.add_argument("-s2", "--scout_2")
    parser.add_argument("-s3", "--scout_3")
    parser.add_argument("-s4", "--scout_4")
    parser.add_argument("-g1", "--guard_1")
    parser.add_argument("-g2", "--guard_2")
    parser.add_argument("-g3", "--guard_3")
    parser.add_argument("-g4", "--guard_4")
    parser.add_argument("-n", "--no_of_matches", type=int, default=1000)
    parser.add_argument("--spectate", action='store_true')
    args = parser.parse_args()

    scout_managers = [
        getattr(import_module(f"{args.scout_1}.rl_manager"), "RLManager"),
        getattr(import_module(f"{args.scout_2}.rl_manager"), "RLManager"),
        getattr(import_module(f"{args.scout_3}.rl_manager"), "RLManager"),
        getattr(import_module(f"{args.scout_4}.rl_manager"), "RLManager"),
    ]
    guard_managers = [
        getattr(import_module(f"{args.guard_1}.rl_manager"), "RLManager"),
        getattr(import_module(f"{args.guard_2}.rl_manager"), "RLManager"),
        getattr(import_module(f"{args.guard_3}.rl_manager"), "RLManager"),
        getattr(import_module(f"{args.guard_4}.rl_manager"), "RLManager"),
    ]

    SAMPLE_SIZE = args.no_of_matches
    TEST_SEEDS = [i for i in range(1, SAMPLE_SIZE+1)]

    CUSTOM_RENDER_MODE = "human" if args.spectate else None
    
    total_scout_rewards = np.zeros((len(scout_managers), SAMPLE_SIZE,))
    total_guard_rewards = np.zeros((len(guard_managers), SAMPLE_SIZE,))

    scout_ep_len = np.zeros((len(scout_managers), SAMPLE_SIZE,))

    for seed_idx in tqdm(range(len(TEST_SEEDS))):
        seed = TEST_SEEDS[seed_idx]

        env = gridworld.env(
            env_wrappers = [],
            render_mode = CUSTOM_RENDER_MODE,
            debug = True,
            novice = False,
            rewards_dict = None,
        )
        env.reset(seed=seed)

        match_scout_rewards = np.zeros((len(scout_managers),))
        match_guard_rewards = np.zeros((len(guard_managers),))

        for round in range(4):
            scout_agents = [RLManager() for RLManager in scout_managers]
            guard_agents = [RLManager() for RLManager in guard_managers]

            for agent in env.agent_iter():
                agent_idx = int(agent[-1])
                observation, reward, termination, truncation, info = env.last()

                if observation["scout"] == 1:
                    match_scout_rewards[agent_idx] += reward
                else:
                    match_guard_rewards[agent_idx] += reward

                if termination or truncation:
                    env.reset()
                    scout_ep_len[agent_idx, seed_idx] = observation["step"] - 1
                    break
                else:
                    if observation["scout"] == 1:
                        action = scout_agents[agent_idx].rl(observation)
                    else:
                        action = guard_agents[agent_idx].rl(observation)

                env.step(action)
        
        total_scout_rewards[:, seed_idx] = match_scout_rewards
        total_guard_rewards[:, seed_idx] = match_guard_rewards

    print(",".join(
        [
                ",".join([
                    str(np.mean(total_scout_rewards[agent_idx])),
                    str(np.var(total_scout_rewards[agent_idx])),
                    str(np.min(total_scout_rewards[agent_idx])),
                    str(np.max(total_scout_rewards[agent_idx])),
                    str(np.mean(scout_ep_len[agent_idx])),
                    str((scout_ep_len == 100).sum()),
                ])
            for agent_idx in range(len(scout_managers))
        ]
        +[
                ",".join([
                    str(np.mean(total_guard_rewards[agent_idx])),
                    str(np.var(total_guard_rewards[agent_idx])),
                    str(np.min(total_guard_rewards[agent_idx])),
                    str(np.max(total_guard_rewards[agent_idx])),
                ])
            for agent_idx in range(len(guard_managers))
        ]
    ))

if __name__ == "__main__":
    main()
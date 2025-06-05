import argparse
from importlib import import_module
from time import sleep

from test_solo_scout import give_env
# from til_environment import gridworld

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scout_name")
    parser.add_argument("--hybrid", action='store_true')
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("-z", "--sleep", action='store_true')
    parser.add_argument("-g", "--guard", type=int, default=0)
    args = parser.parse_args()

    if args.hybrid:
        RLManager = getattr(import_module(f"hybrid.{args.scout_name}.rl_manager"), "RLManager")
    else:
        RLManager = getattr(import_module(f"scouts.{args.scout_name}.rl_manager"), "RLManager")
    model = RLManager()

    # env = gridworld.env(
    #     env_wrappers = [],
    #     render_mode = "human",
    #     debug = True,
    #     novice = False,
    #     rewards_dict = None,
    # )
    TEST_GUARDS = [
        {"main_guard": (-1,-1), "side_guard": (-1,-1)},
        {"main_guard": (1,1), "side_guard": (-1,-1)},
        {"main_guard": (0.375,1), "side_guard": (0.375,1)},
        {"main_guard": (1,1), "side_guard": (1,1)},
    ]
    env_kwargs = TEST_GUARDS[args.guard]
    env = give_env(custom_render_mode="human", **env_kwargs)
    env.reset(seed=args.seed)
    if args.sleep:
        sleep(2)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            if args.sleep:
                sleep(2)
            break
        else:
            action = model.rl(observation)

        env.step(action)

if __name__ == "__main__":
    main()
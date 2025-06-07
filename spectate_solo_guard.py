import argparse
from importlib import import_module
from time import sleep

from til_environment import gridworld
from til_environment.wrappers import GuardEvalWrapper

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("guard_name")
    parser.add_argument("--hybrid", action='store_true')
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("-z", "--sleep", action='store_true')
    parser.add_argument("-g", "--guard", type=int, default=0)
    args = parser.parse_args()

    if args.hybrid:
        RLManager = getattr(import_module(f"hybrid.{args.guard_name}.rl_manager"), "RLManager")
    else:
        RLManager = getattr(import_module(f"guards.{args.guard_name}.rl_manager"), "RLManager")
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
    TEST_SCOUT = getattr(import_module(f"hybrid.stationary.rl_manager"), "RLManager")
    env_kwargs = TEST_GUARDS[args.guard]
    env = give_env(custom_render_mode=None, scout_class=TEST_SCOUT, **env_kwargs)
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
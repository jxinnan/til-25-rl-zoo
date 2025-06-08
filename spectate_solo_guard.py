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
    parser.add_argument("-s", "--scout", type=int, default=0)
    parser.add_argument("-g", "--guard", type=int, default=0)
    args = parser.parse_args()

    if args.hybrid:
        RLManager = getattr(import_module(f"hybrid.{args.guard_name}.rl_manager"), "RLManager")
    else:
        RLManager = getattr(import_module(f"guards.{args.guard_name}.rl_manager"), "RLManager")
    model = RLManager()
    
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
    scout_kwargs = TEST_SCOUTS[args.scout]
    guard_kwargs = TEST_GUARDS[args.guard]
    env = give_env(custom_render_mode="human", **scout_kwargs, **guard_kwargs)
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
import argparse
from importlib import import_module
from time import sleep

from til_environment import gridworld

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scout_name")
    parser.add_argument("--hybrid", action='store_true')
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("-z", "--sleep", action='store_true')
    args = parser.parse_args()

    if args.hybrid:
        RLManager = getattr(import_module(f"hybrid.{args.scout_name}.rl_manager"), "RLManager")
    else:
        RLManager = getattr(import_module(f"scouts.{args.scout_name}.rl_manager"), "RLManager")
    model = RLManager()

    env = gridworld.env(
        env_wrappers = [],
        render_mode = "human",
        debug = True,
        novice = False,
        rewards_dict = None,
    )
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
            if observation["scout"] == 1:
                action = model.rl(observation)
            else:
                action = 4

        env.step(action)

if __name__ == "__main__":
    main()
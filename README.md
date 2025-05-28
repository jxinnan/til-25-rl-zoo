# Zoo :circus_tent:

`uv run test_solo_scout.py [-n NO_OF_MATCHES] [-g TEST_WITH_GUARDS] [--hybrid HYBRID] [-s SAVE_LOG] scout_name`

Check that your imports and filepaths in `rl_manager.py` will work when called from the parent directory. Run with `-n 3` to test if it works.

`uv add` any dependencies you need.

`uv run test_arena.py -s1 scouts.wonder -g1 scouts.wonder -s2 hybrid.cnnppo_v7e1 -g2 hybrid.cnnppo_v7e1 -s3 scouts.atlanta-8M-guard-exp -g3 guards.helvetica -s4 scouts.antwerp-noguard -g4 guards.helvetica -n 2 --spectate`

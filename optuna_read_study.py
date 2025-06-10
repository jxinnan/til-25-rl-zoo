import optuna
import pickle

if __name__ == "__main__":
    # study = optuna.load_study(
    #     study_name="fellowship_optuna2",
    #     storage="mysql+pymysql://root@localhost/fellowship_optuna2",
    # )
    with open("run_hide_tell_study", "rb") as f:
        trials = pickle.load(f)
    trials = [trial for trial in trials if trial.values != None]
    # print(len(trials))
    # print(trials[0].params)
    # print(trials[0].values[0])
    trials = sorted(trials, key = lambda x : x.values[0])

    PRUNE_THRESH = 0.05

    good_trials = []
    i = 0
    while i < len(trials):
        base_trial = trials[i]
        # print(base_trial.values[0])
        if base_trial.values[0] > 0.46:
            break

        base_weights = base_trial.user_attrs
        good_trials.append(base_trial)
        j = i + 1
        while j < len(trials):
            next_weights = trials[j].user_attrs
            prune_count = 0
            for key in base_weights:
                if abs(base_weights[key] - next_weights[key]) <= PRUNE_THRESH:
                    prune_count += 1
            if prune_count == len(base_weights):
                trials.pop(j)
            else:
                j += 1
        i += 1
    
    good_weights = [trial.user_attrs for trial in good_trials]
    
    for trial in good_trials:
        for key in trial.user_attrs:
            # if trial.user_attrs[key] < 0.05:
            #     print(0, end=", ")
            # else:
            print(trial.user_attrs[key], end=", ")
        # print(trial.values[0])
        print()
        print()
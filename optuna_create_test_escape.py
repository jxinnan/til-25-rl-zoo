import optuna

if __name__ == "__main__":
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),
        storage="mysql+pymysql://root@localhost/run_hide_tell",
        study_name="run_hide_tell",
    )

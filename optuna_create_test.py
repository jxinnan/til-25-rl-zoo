import optuna

if __name__ == "__main__":
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),
        storage="mysql+pymysql://root@localhost/fellowship_optuna2",
        study_name="fellowship_optuna2",
    )

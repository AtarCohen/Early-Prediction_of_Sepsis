import optuna
from LSTM_main import main

if __name__ == '__main__':
    # HPT_Tune_1
    study = optuna.create_study(study_name='HPT_split_0', direction='maximize')
    # study = optuna.create_study(study_name = 'debug', direction='maximize')
    study.optimize(main)
    # main()  # uncomment to run normaly
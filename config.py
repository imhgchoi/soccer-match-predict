import argparse


def get_args():
    argp = argparse.ArgumentParser(description='Soccer Match Prediction',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--debug', action='store_true', default=False)

    # Direcotories
    argp.add_argument('--datadir', type=str, default="./data/")

    # Data
    argp.add_argument('--use_prepro_v1', action='store_true', default=False)
    argp.add_argument('--test_years', nargs='+', type=int, default=[2018])
    argp.add_argument('--lookback_opp_matches', type=int, default=3)   # how many matches between away and home to refer to
    argp.add_argument('--lookback_matches', type=int, default=5)   # how many previous matches of a team to refer to

    # Model General
    argp.add_argument('--model_type', type=str, default='linreg', choices=['linreg'])

    # linear regression
    argp.add_argument('--linreg_w_init', type=str, default='uniform', choices=['uniform','xavier'])
    argp.add_argument('--linreg_alpha', type=float, default=0.01)
    argp.add_argument('--linreg_tolerance', type=float, default=1e-7)

    # evaluation
    argp.add_argument('--eval_metrics', nargs='+', type=str, default=['mse','mae'], choices=['mse','mae'])


    return argp.parse_args()
import argparse


def get_args():
    argp = argparse.ArgumentParser(description='Soccer Match Prediction',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--debug', action='store_true', default=False)

    # Directories
    argp.add_argument('--datadir', type=str, default="./data/")

    # Data
    argp.add_argument('--use_preprocessed', action='store_true', default=False)
    argp.add_argument('--test_years', nargs='+', type=int, default=[2018])
    argp.add_argument('--lookback_opp_matches', type=int, default=3)   # how many matches between away and home to refer to
    argp.add_argument('--lookback_matches', type=int, default=5)   # how many previous matches of a team to refer to

    # Model General
    argp.add_argument('--model_type', type=str, default='linreg', choices=['linreg','ridge','lasso','logreg','svm','nnreg',
                                                                           'nnclass','kernelreg','knnreg','knnclass'])

    # linear regression
    argp.add_argument('--linreg_w_init', type=str, default='uniform', choices=['uniform','xavier'])
    argp.add_argument('--linreg_alpha', type=float, default=0.01)
    argp.add_argument('--linreg_tolerance', type=float, default=1e-7)

    # KNN classification
    argp.add_argument('--knn_k', type=int, default=11)

    # evaluation
    argp.add_argument('--eval_metrics', nargs='+', type=str, default=['mse','mae'], choices=['mse','mae'])
    argp.add_argument('--bet_money', type=int, default=5000)
    argp.add_argument('--commission', type=float, default=0.0)
    argp.add_argument('--tax', type=float, default=0.0)


    return argp.parse_args()
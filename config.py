import argparse


def get_args():
    argp = argparse.ArgumentParser(description='Soccer Match Prediction',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--debug', action='store_true', default=False)

    # Direcotories
    argp.add_argument('--datadir', type=str, default="./data/")

    # Data
    argp.add_argument('--test_years', nargs='+', type=int, default=[2018])

    # Model General
    argp.add_argument('--model_type', type=str, default='linreg', choices=['linreg'])

    # linear regression
    argp.add_argument('--w_init', type=str, default='uniform', choices=['uniform','xavier'])
    argp.add_argument('--alpha', type=float, default=0.01)



    return argp.parse_args()
import argparse


def get_argument_train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help='Print debug traces.',
        action='store_true')
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=-1)
    parser.add_argument(
        '-n', '--noise',
        help='Add noise to the data input randomly',
        default=0)
    args = parser.parse_args()
    return args


args = get_argument_train()

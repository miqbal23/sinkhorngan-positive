import argparse
import multiprocessing
import random
import sys

import torch

from aggregator import Session

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if sys.version_info[0] < 3:
    raise Exception("Please use Python version 3.6")
elif sys.version_info.major > 3 and sys.version_info.minor < 6:
    raise Exception("Please use Python version 3.6 or use manual_main.py (define by your self)")

parser = argparse.ArgumentParser()
# Train mode
parser.add_argument("--train", action='store_const', dest='action', const='train', help="for Train mode")
parser.add_argument("--eval", action='store_const', dest='action', const='eval', help="eval a bunch of model")
parser.add_argument('-c', "--config", type=str, default=None, help="config path or an session dir for continuing the training")
parser.add_argument("--only_last", action='store_const', dest='only_last', const=True, default=False, help="assign if only eval last model")
parser.add_argument("--debug", action='store_const', dest='debug', const=True, default=False, help="add if run in debuggin mode")
args = parser.parse_args()

if args.debug:
    multiprocessing.set_start_method('spawn', True)


def main():
    assert args.config, "Please input Config path or session dir !"
    assert args.action, "Please select mode, train or eval !"
    sess = Session(args.config)
    if args.action == "train":
        sess.train()
    elif args.action == 'eval':
        sess.eval(args.only_last)


if __name__ == "__main__":
    main()

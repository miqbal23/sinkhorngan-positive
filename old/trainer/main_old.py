# to run your script code just using
# python manage.py train Ganomaly.py
import multiprocessing
import sys

if sys.version_info[0] < 3:
    raise Exception("Please use Python version 3.6")
elif sys.version_info.major > 3 and sys.version_info.minor < 6:
    raise Exception("Please use Python version 3.6 or use manual_main.py (define by your self)")

import argparse
from importlib import import_module
import torch
from _old.dataloader import datasets_dict

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('network', type=str, default="mlp", help="network you want to run")
parser.add_argument('dataset', type=str, default="mnist", help="dataset you want to run, example, {datasets}".format(
    datasets="|".join(datasets_dict.keys())))
parser.add_argument('--train', action='store_const', dest='action', const='train', help="Set if train mode")
parser.add_argument('--eval', action='store_const', dest='action', const='eval', help="Set if train mode")
parser.add_argument('--debug', action='store_const', dest='action', const='debug', help="Set if train mode")
parser.add_argument('--pre_train_dir', type=str, default=None, help="add pretrain model if exist")
parser.add_argument('--n_gpu', type=int, default=1, help="Number GPU Use")
parser.add_argument('-d', "--dir", type=str, default="../experiments", help="select saved logs and model directory")
args = parser.parse_args()

PARAMS = {"n_gpu": args.n_gpu,
          "network": {
              "z_dim": 100,
              "ngf": 64,
              "ndf": 64,
              "n_extra_layers": 0,
              "output_disc": 100},
          "train": {
              "save_period": 20,
              "check_generated_data": True,
              "generated_size": 8 * 8,
              "n_critic": 5,
              "n_iter_D": 2,
              "n_iter_G": 1},
          "optimizer": {
              "lr_Gen": 1e-4,
              "lr_Dis": 1e-4,
              "betas": (0.5, 0.999)},
          "regularization": {
              "clipping": None,
          },
          "criterion": {},
          "dataloader": datasets_dict[args.dataset],
          "WGAN": {
              "gradient_penalty": True,
              "w_div": False,
              "lambda_gp": 0.4,
              "clipping": None},
          "MMD": {
              "clipping": None},
          "Sinkhorn": {
              "clipping": 0.01,
              "margin": 5}
          }

data_loader = {
    "shuffle": True,
    "sampler": None,
    "batch_sampler": None,
    "num_workers": 2,  # default 4
    "collate_fn": None,
    "pin_memory": False,
    "drop_last": False,
    "timeout": 0,
    "worker_init_fn": None,
    "multiprocessing_context": None
}

batch_size_list = [200]  #
niter_list = [10, 100, 1000]  # default 10
epsilon_list = [0.01, 1000, 10000, 100000]
epochs = 100000

if args.action == "debug":
    multiprocessing.set_start_method('spawn', True)
    batch_size_list = [8]


def main():
    # load module
    network_name = args.network.split('.')[0] if ".py" in args.network else args.network
    PARAMS["network"]["name"] = network_name
    try:
        network = import_module("networks.{}".format(network_name))
    except ImportError as err:
        raise UserWarning("Error importing {}: {}".format(args.network, err))

    if args.action.lower() == "train" or args.action.lower() == "debug":
        exp_number = 0
        for batch_size in batch_size_list:
            exp_number += 1
            # # # MMD ..................................................................................
            # mmd = MMD(PARAMS, network, args.pre_train_dir)
            # mmd.train(epochs, batch_size, **data_loader)
            # del mmd

            # # # WGAN ..................................................................................
            # wgan = WGAN(PARAMS, network, args.pre_train_dir)
            # wgan.train(epochs, batch_size, **data_loader)
            # del wgan

            # # RAWGAN ..................................................................................
            # rawgan = RAWGAN(PARAMS, network, args.pre_train_dir)
            # rawgan.train(epochs, batch_size, **data_loader)
            # del rawgan

            # # RAGAN
            # ragan = RAGAN(PARAMS, network, args.pre_train_dir)
            # ragan.train(epochs, batch_size, **data_loader)
            # del ragan

            for n_s in niter_list:
                for epsilon in epsilon_list:
                    other_note = ""
                    PARAMS["Sinkhorn"]["niter_sink"] = n_s
                    PARAMS["Sinkhorn"]["epsilon"] = epsilon
                    PARAMS["experiment_name"] = "batch{batch}_iter{iter}_eps{eps}{other}".format(
                        eps=epsilon, iter=n_s, batch=batch_size // 2, other=other_note)
                    exp_number += 1

                    # NSSinkhornSingle ................................................................................
                    nssinkhornsingle = NSSinkhornSingle(PARAMS, network, args.pre_train_dir)
                    nssinkhornsingle.train(batch_size, **data_loader)
                    del nssinkhornsingle

                    # SNSinkhorn multiplebatch  ................................................................................
                    # nssinkhornmultiple = NSSinkhornMultiple(PARAMS, network, args.pre_train_dir)
                    # nssinkhornmultiple.train(epochs, batch_size, **data_loader)
                    # del nssinkhornmultiple

                    # # Sinkhorn primal ................................................................................
                    # sinkhorn = Sinkhorn(PARAMS, network, args.pre_train_dir)
                    # sinkhorn.train(epochs, batch_size, **data_loader)
                    # del sinkhorn

                    # # Sinkhorn_EOT primal ................................................................................
                    # sinkhorn = Sinkhorn_EOT(PARAMS, network, args.pre_train_dir)
                    # sinkhorn.train(epochs, batch_size, **data_loader)
                    # del sinkhorn


    elif args.action.lower() == "eval":
        # MMD ..................................................................................
        mmd = MMD(PARAMS, network)
        mmd.evaluate()
        # WGAN ..................................................................................
        wgan = WGAN(PARAMS, network)
        wgan.evaluate()
        # Sinkhorn  ................................................................................
        sinkhorn = Sinkhorn(PARAMS, network)
        sinkhorn.evaluate()
    else:
        raise UserWarning("Please select the action <train> or <eval>")
    pass


if __name__ == "__main__":
    main()

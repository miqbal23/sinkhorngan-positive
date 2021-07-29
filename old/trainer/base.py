import json
import os
import time
from abc import ABC

import numpy as np
import torch
import torchvision
from modules.trainer.regularization import weight_clipping
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from modules.evaluator import Evaluation
from utils import makedir_if_not_exist


class TrainerBase(ABC):
    """
    Basic abstract to train your model, you need override this abstract and set the params
    """

    list_loss = []
    fixed_samples = {}
    tensorboard_scalar = {}
    tensorboard_histogram = {}

    global_step = 0
    evaluation = None

    # initial Tensor Type
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    Eval_net = None

    one = FloatTensor([1])
    mone = one * -1
    epoch = 0

    def __init__(self, params, network, pre_train=None):
        # Init id
        self.date_train = time.asctime()

        self.params = params.copy()

        # Init Network and DataParallel ..............................................................................
        self.models = MultipleModel(
            Dis=network.Discriminator(self.params),
            Gen=network.Generator(self.params)
        )

        if pre_train:
            self.networks_load(pre_train, eval=False, check_result=False)

        # Set Parallel Train
        if torch.cuda.is_available() and self.params["n_gpu"] > 1:
            self.models.set_parallel(device_ids=range(self.params["n_gpu"]))

        # Set network eval
        self.Eval_net = "Gen"

        # Init number iter each network
        self.n_critic = self.params["train"]["n_critic"]

        # Init dir
        self.init_dir()

        # Init Datasets  ................................................................................
        if self.params["dataloader"]["type"] is not None:
            self.dataset = Datasets(is_train=True, **self.params)

        # Init writer to tensorboard, and add configuration text
        self.writer = SummaryWriter(self.params["dir"])
        self.writer.add_text("parameters", json.dumps(self.params))

        # Init eval
        if "eval" in self.params.keys():
            if self.params["evaluator"]["type"] is not None:
                self.evaluation = Evaluation(self.writer, **self.params)

    def init_dir(self):
        """
        init all directory
        :return:
        """
        # reset root_dir = root_dir/experiments
        try:
            self.params["root_dir"]
        except KeyError:
            self.params["root_dir"] = ".."

        self.params["root_dir"] = os.path.join(self.params["root_dir"], "experiments")
        utils.print_wline("Initiate all directories", "=")
        makedir_if_not_exist(self.params["root_dir"])

        # set dataloader dir = root_dir/experiments/dataloader
        self.params["dataloader"]["dir"] = os.path.join(self.params["root_dir"], "dataloader")
        makedir_if_not_exist(self.params["dataloader"]["dir"])

        # set train dir =  root_dir/experiments/train/dataloader/train_mode/model/experiment
        experiment_dir = "train/{datasets}/{train_mode}/{network}/{experiment}".format(
            datasets=self.params['dataloader']['type'],
            train_mode=self.__class__.__name__,
            network=self.params["network"]["name"],
            experiment=self.params["experiment_name"]
        )

        # set experiment dir and logs, if exist add number on post
        _dir = os.path.join(self.params["root_dir"], experiment_dir)
        self.params["dir"] = _dir
        _copy = 1
        while os.path.exists(self.params["dir"]):
            self.params["dir"] = "{dir}_{cp}".format(dir=_dir, cp=_copy)
            _copy += 1

        makedir_if_not_exist(self.params["dir"])

        # set save models dir = root_dir/experiments/train/dataloader/train_mode/model/case/saved_model
        self.params["train"]["saved_model_dir"] = os.path.join(self.params["dir"], "saved_model")
        makedir_if_not_exist(self.params["train"]["saved_model_dir"])

        _print = [" - Root       : {}/...".format(os.path.abspath(self.params["root_dir"])),
                  " - Datasets   : {}".format(os.path.abspath(self.params["dataloader"]["dir"])),
                  " - Session    : {}".format(os.path.abspath(self.params["dir"])),
                  " - Save Model : {}".format(os.path.abspath(self.params["train"]["saved_model_dir"]))]

        print('\n'.join(_print))

    def forward(self, **kwargs):
        with torch.no_grad():
            samples = self.models[self.Eval_net](z_input=kwargs["z_input"]).detach()
        return samples

    # Training ##############################################################################
    def pre_train(self):
        """
        For adding some process before training session
        :return:
        """
        pass

    def post_train(self):
        """
        For adding some process after training session
        :return:
        """
        pass

    def train(self, epochs, batch_size, **kwargs):
        """
        :param epochs:
        :param batch_size:
        :return:
        """
        self.global_step = 0

        self.models.cuda()

        if "eval" in self.params.keys():
            if self.Eval_net is None and self.params["evaluator"]["type"] is not None:
                raise ValueError('Please define your eval network in self.Eval_net')

        # save network architecture on tensorboard
        self.tensorboard_add_networks()

        utils.print_wline("Script for Check log".format(epochs), line="=")
        print("tensorboard --logdir {} --bind_all".format(os.path.abspath(self.params["dir"])))

        utils.print_wline("Start Train with {} Epoch".format(epochs), line="=")

        # Init data loader
        if torch.cuda.is_available() and self.params["n_gpu"] > 1:
            batch_size = batch_size * self.params["n_gpu"]

        self.pre_train()

        data_loader = DataLoader(self.dataset.data, batch_size=batch_size, **kwargs)

        for self.epoch in range(self.epoch + 1, epochs + 1):

            self.models.train()
            for idx, (x, y) in enumerate(
                    tqdm(data_loader, desc="Epoch {epoch}/{epochs}".format(epoch=self.epoch, epochs=epochs))):
                if idx == data_loader.dataset.__len__() // batch_size:
                    break

                x = x.type(self.FloatTensor)
                y = y.type(self.FloatTensor)

                # Update parameters
                self.update_parameters(idx, x, y)

                self.global_step += 1

            self.tensorboard_update(global_step=self.epoch, pos_tag="Epoch_")
            # check trained data
            if self.params["train"]["check_generated_data"]:
                self.check_generated_data(self.fixed_samples, global_step=self.epoch, tag="result")

            # save each self.params["train"]["save_period"] epoch
            if self.epoch % self.params["train"]["save_period"] == 0 and self.epoch != epochs:
                self.networks_save()

        if "eval" in self.params.keys():
            self.evaluate()

        self.post_train()
        self.networks_save()
        utils.print_wline("Train with {} Epoch Done !".format(self.epoch + 1), line="=")

    def update_parameters(self, idx, x, y):
        """
        This is the main Update parameters
        :param idx: idx
        :param x:
        :param y:
        :return:
        """
        d_loss = []
        g_loss = []

        if self.epoch < 25 or self.epoch % 500 == 0:
            self.n_critic = 20
        else:
            self.n_critic = 5

        z = self.FloatTensor(np.random.normal(0, 1, (x.shape[0], self.params["network"]["z_dim"])))
        self.models.zero_grad("Dis")
        d_loss.append(self.update_parameters_discriminator(z, x, y).item())
        self.models.step("Dis")

        # Clipping D
        if self.params["regularization"]["clipping"]:
            weight_clipping(self.models["Dis"], self.params["regularization"]["clipping"])

        self.models.zero_grad("Gen")
        if idx % self.n_critic == 0:
            g_loss.append(self.update_parameters_generator(z, x, y).item())
            self.models.step("Gen")

            _scalar = {
                "Gen_loss": np.mean(g_loss),
                "Dis_loss": np.mean(d_loss)
            }

            for name, value in _scalar.items():
                if name not in self.tensorboard_scalar:
                    self.tensorboard_scalar[name] = [value]
                else:
                    self.tensorboard_scalar[name].append(value)

            # self.tensorboard_update(self.global_step, pos_tag="Step_")

    def update_parameters_generator(self, z, x, y):
        """
        Optional use
        :param idx:
        :param z:
        :param x:
        :param y:
        :return:
        """
        raise ValueError('Please put update generator')

    def update_parameters_discriminator(self, z, x, y):
        """
        Optional use
        :param idx:
        :param z:
        :param x:
        :param y:
        :return:
        """
        raise ValueError('Please put update discriminator')

    # Evaluation ##############################################################################

    def check_generated_data(self, samples, global_step=0, tag="result"):
        """
        this function used for check the result of generator network and save it to tensorboard
        :param global_step:
        :param samples(dict): samples of input network
        :param tag: save the output to tensorboard log wit tag
        :return:
        """
        self.models.eval()
        images = self.forward(**samples)

        self.tensorboard_add_images(images, global_step, tag)

    def evaluate(self):
        self.evaluation.run(self.models[self.Eval_net], self.global_step)

    # Tensorboard #############################################################################

    def tensorboard_update(self, global_step=0, pos_tag=""):
        """
        save all self.tensorboard_* to tensorboard log
        :return:
        """

        if len(self.tensorboard_scalar.keys()) != 0:
            for key, value in self.tensorboard_scalar.items():
                if isinstance(value, list):
                    value = np.mean(value, 0)
                self.writer.add_scalar(tag=pos_tag + key, scalar_value=value, global_step=global_step,
                                       walltime=time.time())

        if len(self.tensorboard_histogram.keys()) != 0:
            for key, value in self.tensorboard_histogram.items():
                value = np.array(value)
                if len(value.shape) > 1:
                    value = np.mean(value, 0)
                self.writer.add_histogram(tag=pos_tag + key, values=value, global_step=global_step,
                                          walltime=time.time())

        # Clean Dictionary
        self.tensorboard_scalar.clear()
        self.tensorboard_histogram.clear()

    def tensorboard_add_images(self, images, global_step, tag):
        if torch.cuda.is_available():
            images = images.cpu()
        else:
            images = images

        grid = torchvision.utils.make_grid(images, normalize=True)
        self.writer.add_image(tag, grid, global_step, walltime=time.time())

    def tensorboard_add_networks(self):
        for name, model in self.models.items():
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            self.writer.add_text("Network {name}-{num_params}params".format(name=name, num_params=num_params),
                                 str(model))

    # Networks Utils ##########################################################################

    def networks_save(self, output_ext="pkl"):
        """
        save model to root_dir/type/name/saved_model
        :param output_ext:
        :return:
        """

        more_info = {
            "dataset": self.params['dataloader']['type'],
            "train_mode": self.__class__.__name__
        }

        self.models.save(path=self.params["train"]["saved_model_dir"], global_step=None, more_info=more_info,
                         output_ext=output_ext)

        print('model saved! [epoch:{epoch}]'.format(epoch=self.epoch))

    def networks_load(self, dir, eval=False, check_result=False):
        """
        :param path: path model
        :return:
        """
        utils.print_wline("Load Network", line="=")

        self.models.load(dir)
        self.epoch = self.models.global_step

        if eval:
            self.evaluate()
        if check_result:
            self.check_generated_data(self.fixed_samples, tag="Pre-trained_Model_Result")

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from base import BaseEvaluator
from modules.evaluator.FID.fid_score import calculate_metric
from modules.evaluator.FID.inception import InceptionV3


class FID(BaseEvaluator):
    def __init__(self, parent, name=None, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        self.dim_activation = 2048

    def eval(self, step, tag=None, run_per=None, **kwargs):
        # Data Usage
        assert self.configs["total_data"] % self.configs["batch_size"] == 0, "Make sure total_data can devided by batch_size"
        n_batches = self.configs["total_data"] // self.configs["batch_size"]

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dim_activation]
        network_eval = InceptionV3([block_idx])
        network_generator = self.model['generator']
        data_loader = self.dataloader['main'](batch_size=self.configs["batch_size"])
        z_mean = self.configs.get('mean', 0)
        z_std = self.configs.get('std', 1)

        if torch.cuda.is_available():
            network_generator.cuda()
            network_eval.cuda()

        network_eval.eval()
        network_generator.eval()

        prediction_real = np.empty((self.configs["total_data"], self.dim_activation))
        prediction_fake = np.empty((self.configs["total_data"], self.dim_activation))

        def get_prediction(data):
            with torch.no_grad():
                _predictions = network_eval(data)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if _predictions.shape[2] != 1 or _predictions.shape[3] != 1:
                _predictions = F.adaptive_avg_pool2d(_predictions, output_size=(1, 1))
            return _predictions.cpu().data.numpy().reshape(self.configs["batch_size"], -1)

        # iterate over all data
        for idx in tqdm(range(n_batches), desc="Evaluate Fréchet Inception Distance"):
            x, y = next(iter(data_loader))
            start = idx * self.configs["batch_size"]
            end = start + self.configs["batch_size"]
            # set input whether use cuda or not
            x = x.to(self.device)
            ###############################################
            # Calculate Real Image
            ###############################################
            prediction_real[start:end] = get_prediction(x)
            ###############################################
            # Calculate Fake Image
            ###############################################
            z_dim = (x.shape[0], self.model.configs['generator']['z_dim'])
            z = self.Tensor.randn(*z_dim, mean=z_mean, std=z_std)
            with torch.no_grad():
                fake_images = network_generator(z).detach()
            prediction_fake[start:end] = get_prediction(fake_images)

        # calculate mean and covariance
        mu_real = np.mean(prediction_real, axis=0)
        sigma_real = np.cov(prediction_real, rowvar=False)

        # calculate mean and covariance
        mu_fake = np.mean(prediction_fake, axis=0)
        sigma_fake = np.cov(prediction_fake, rowvar=False)

        score = calculate_metric(mu_real, sigma_real, mu_fake, sigma_fake)

        if tag is None:
            tag = "Fréchet Inception Distance Score"
        _tag = "{tag}.___per_{n}_{typ}".format(tag=tag, n=run_per[0], typ=run_per[1]) if run_per else tag
        self.writer.add_scalar(_tag, score, global_step=step)
        del network_eval
        del data_loader

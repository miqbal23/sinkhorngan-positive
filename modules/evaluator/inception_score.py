import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import inception_v3
from tqdm import tqdm

from base import BaseEvaluator


class IS(BaseEvaluator):
    def __init__(self, parent, name=None, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        self.dim_activation = 1000

    def eval(self, step, tag=None, run_per=None, **kwargs):
        # Data Usage
        assert self.configs["total_data"] % self.configs["batch_size"] == 0, "Make sure total_data can devided by batch_size"
        n_batches = self.configs["total_data"] // self.configs["batch_size"]

        network_eval = inception_v3(pretrained=True)
        network_generator = self.model['generator']
        z_mean = self.configs.get('mean', 0)
        z_std = self.configs.get('std', 1)

        if torch.cuda.is_available():
            network_generator.cuda()
            network_eval.cuda()

        network_eval.eval()
        network_generator.eval()
        up = nn.UpsamplingBilinear2d(size=(299, 299))
        scores = []
        for idx in tqdm(range(n_batches), desc="Evaluate Inception Score"):
            z_dim = (self.configs["batch_size"], self.model.configs['generator']['z_dim'])
            z = self.Tensor.randn(*z_dim, mean=z_mean, std=z_std)
            with torch.no_grad():
                fake_images = network_generator(z)
                if fake_images.shape[-1] != 299:
                    fake_images = up(fake_images)
                x = network_eval(fake_images).detach()
                scores.append(x)
        p_yx = F.softmax(torch.cat(scores, 0), 1)
        p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
        KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
        final_score = KL_d.mean()

        if tag is None:
            tag = "Inception Score"
        _tag = "{tag}.___per_{n}_{typ}".format(tag=tag, n=run_per[0], typ=run_per[1]) if run_per else tag
        self.writer.add_scalar(_tag, final_score, global_step=step)
        del network_eval

import time

import torch
import torchvision

from base import BaseEvaluator


class GeneratedImage(BaseEvaluator):
    def __init__(self, parent, name=None, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        z_mean = self.configs.get('mean', 0)
        z_std = self.configs.get('std', 1)
        self._data = self.Tensor.randn(self.configs['total_data'], self.configs['z_dim'], mean=z_mean, std=z_std)

    def eval(self, step, tag=None, run_per=None, **kwargs):
        self.model.eval()
        with torch.no_grad():
            images = self.model['generator'](self._data)
        if torch.cuda.is_available():
            images = images.cpu()
        else:
            images = images

        grid = torchvision.utils.make_grid(images, normalize=True)

        if tag is None:
            tag = "Generated Image"
        _tag = "{tag}.___per_{n}_{typ}".format(tag=tag, n=run_per[0], typ=run_per[1]) if run_per else tag
        self.writer.add_image(_tag, grid, step, walltime=time.time())

import torch
from pytorch_msssim import ssim # ms_ssim

from base import BaseEvaluator


class MsSsim(BaseEvaluator):
    def __init__(self, parent, name=None, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        z_mean = self.configs.get('mean', 0)
        z_std = self.configs.get('std', 1)
        self._data = self.Tensor.randn(self.configs['total_data'], self.configs['z_dim'], mean=z_mean, std=z_std)
        self.result_before = {'generator': None, 'critic': None}

    def eval(self, step, tag=None, run_per=None, **kwargs):
        tag = "" if tag is None else tag
        self.model.eval()
        with torch.no_grad():
            images = self.model['generator'](self._data)

        if torch.cuda.is_available():
            images = images.cpu()

        # Check Generator ssim and ms_ssim ############################
        if self.result_before['generator'] is None:
            ssim_score = 0
            ms_ssim_loss = 0
        else:
            # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
            # normalize image from [-1, 1] => [0, 255]
            x = (self.result_before['generator'] + 1) * 127.5
            y = (images + 1) * 127.5
            ssim_score = ssim(x, y, data_range=255, size_average=True)
#             ms_ssim_loss = 1 - ms_ssim(x, y, data_range=255, size_average=True)

        tag_g = tag + "SSIM Generator"
        _tag = "{tag}.___per_{n}_{typ}".format(tag=tag_g, n=run_per[0], typ=run_per[1]) if run_per else tag_g
        self.writer.add_scalar(_tag, ssim_score, global_step=step)

#         tag_g = tag + "MS_SSIM Generator"
#         _tag = "{tag}.___per_{n}_{typ}".format(tag=tag_g, n=run_per[0], typ=run_per[1]) if run_per else tag_g
#         self.writer.add_scalar(_tag, ms_ssim_loss, global_step=step)

        self.result_before['generator'] = images.cpu()

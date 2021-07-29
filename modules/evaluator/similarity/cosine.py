import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from base import BaseEvaluator


class CosineSimilarity(BaseEvaluator):
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
            critic = self.model['critic'](images)

        if torch.cuda.is_available():
            images = images.cpu()
            critic = critic.cpu()

        # Check Critic ############################
        if self.result_before['critic'] is None:
            score = 0
        else:
            # calculate similarity
            score = cosine_similarity(self.result_before['critic'], critic).mean()

        tag_c = tag+"Cosine Similarity Critic"
        _tag = "{tag}.___per_{n}_{typ}".format(tag=tag_c, n=run_per[0], typ=run_per[1]) if run_per else tag_c
        self.writer.add_scalar(_tag, score, global_step=step)

        # Check Generator ############################
        if self.result_before['generator'] is None:
            score = 0
        else:
            # calculate similarity
            # Create m x d data matrix
            m = len(images)
            d = np.product(images.shape[1:])
            score = cosine_similarity(np.reshape(self.result_before['generator'], (m, d)),
                                      np.reshape(images, (m, d))).mean()

        tag_g = tag + "Cosine Similarity Generator"
        _tag = "{tag}.___per_{n}_{typ}".format(tag=tag_g, n=run_per[0], typ=run_per[1]) if run_per else tag_g
        self.writer.add_scalar(_tag, score, global_step=step)

        self.result_before['critic'] = critic.cpu().numpy()
        self.result_before['generator'] = images.cpu().numpy()

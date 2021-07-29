#!/usr/bin/env python3
"""Calculates the Frechet IS Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from scipy import linalg

# from scipy.misc import imread

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


def calculate_metric(mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6):
    """
    Evaluate FID score given two pairs of means and covariances from two different image sets

    Parameters
    ----------
    mu_real, sigma_real : float
        Mean (mu) and covariance (sigma) of real or reference images set
    mu_fake, real_fake : float
        Mean (mu) and covariance (sigma) of images set to be tested

    Returns
    -------
    fid_score : float
        Score of Frechet Inception Distance between two image sets
        :param eps:
        :param mu_real:
        :param sigma_real:
        :param mu_fake:
        :param sigma_fake:
    """
    mu1 = np.atleast_1d(mu_real)
    mu2 = np.atleast_1d(mu_fake)

    sigma1 = np.atleast_2d(sigma_real)
    sigma2 = np.atleast_2d(sigma_fake)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

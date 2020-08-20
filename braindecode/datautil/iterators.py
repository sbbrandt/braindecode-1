"""Custom iterator for training epochs
"""

# Authors: Simon Brandt <simonbrandt@protonmail.com>
#
# License: BSD (3-clause)
 
import numpy as np
import torch


class mixup_iterator(torch.utils.data.DataLoader):
    """Implements Iterator for Mixup for EEG data. See [mixup].

    Code adapted from
    https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py

    Parameters
    ----------
    dataset: Dataset
        dataset from which to load the data.

    alpha: float
        mixup hyperparameter.

    beta_per_sample: bool (default=False)
        by default, one mixing coefficient per batch is drawn from an beta
        distribution. If True, one mixing coefficient per sample is drawn.

    References
    ----------
    ..  [mixup] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
        mixup: Beyond Empirical Risk Minimization
        Online: https://arxiv.org/abs/1710.09412

    """
    def __init__(self, dataset, alpha, beta_per_sample=False, **kwargs):
        self.alpha = alpha
        self.beta_per_sample = beta_per_sample
        super().__init__(dataset, collate_fn=self.mixup, **kwargs)

    def mixup(self, data):
        batch_size = len(data)
        n_channels, n_times = data[0][0].shape

        if self.alpha > 0:
            if self.beta_per_sample:
                lam = np.random.beta(self.alpha, self.alpha, batch_size)
            else:
                lam = np.ones(batch_size)
                lam *= np.random.beta(self.alpha, self.alpha)
        else:
            lam = np.ones(batch_size)

        idx_perm = np.random.permutation(batch_size)
        x = np.zeros((batch_size, n_channels, n_times))
        y_a = np.arange(batch_size)
        y_b = np.arange(batch_size)
        supercrop_inds = np.zeros((batch_size, 3))

        for idx in range(batch_size):
            x[idx] = lam[idx] * data[idx][0] \
                     + (1 - lam[idx]) * data[idx_perm[idx]][0]
            y_a[idx] = data[idx][1]
            y_b[idx] = data[idx_perm[idx]][1]
            supercrop_inds[idx] = np.array(data[idx][2])

        x = torch.tensor(x).type(torch.float32)
        y_a = torch.tensor(y_a).type(torch.int64)
        y_b = torch.tensor(y_b).type(torch.int64)
        lam = torch.tensor(lam).type(torch.float32)
        supercrop_inds = torch.tensor(supercrop_inds).type(torch.int64)

        supercrop_inds = list(supercrop_inds.T)

        return x, (y_a, y_b, lam), supercrop_inds

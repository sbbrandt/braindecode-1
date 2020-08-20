import torch


class CroppedLoss:
    """Compute Loss after averaging predictions across time.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)"""

    def __init__(self, loss_function):
        self.loss_function = loss_function

    def __call__(self, preds, targets):
        avg_preds = torch.mean(preds, dim=2)
        avg_preds = avg_preds.squeeze(dim=1)
        return self.loss_function(avg_preds, targets)

class mixup_criterion:
    """Implements loss for Mixup for EEG data. See [mixup].

    Code adapted from
    https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py

    Parameters
    ----------
    preds: torch tensor
        predictions from the model

    target: torch tensor | list of torch tensor
        For predictions without mixup, the targets as a tensor. If mixup has
        been applied, a list containing the targets of the two mixed
        samples and the mixing coefficients as tensors.


    References
    ----------
    ..  [mixup] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
        mixup: Beyond Empirical Risk Minimization
        Online: https://arxiv.org/abs/1710.09412

    """

    def __call__(self, preds, target):
        return mixup_criterion.loss_function(preds, target)

    @staticmethod
    def loss_function(preds, target):
        if len(target) == 3:
            # unpack target
            y_a, y_b, lam = target
            # compute loss per sample
            loss_a = torch.nn.functional.nll_loss(preds,
                                                  y_a,
                                                  reduction='none')
            loss_b = torch.nn.functional.nll_loss(preds,
                                                  y_b,
                                                  reduction='none')
            # compute weighted mean
            ret = torch.mul(lam, loss_a) + torch.mul(1 - lam, loss_b)
            return ret.mean()
        else:
            return torch.nn.functional.nll_loss(preds,
                                                target)

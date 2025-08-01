from .. import base, cfg
from .phase1 import *

np = base.np


class Loss:
    """
    default: MSE\n
    because of ReNNe.Optimizer is a invert gradient descent optimizer\n
    so the gradient of output is the negative of the gradient of loss\n
    """

    def __init__(self, net):
        self.net: base.Net = net

    def lossAnArray(self, y, y_dst):
        return np.sum((y_dst - y) ** 2)

    def deriv_lossAnArray(self, y, y_dst):
        return 2 * (y_dst - y)

    def evaluate(self, y, y_dst):
        loss = 0.0
        accuracy = 0.0
        total_elementcnt = 0
        for yi, yi_dst in zip(y, y_dst):
            loss += self.lossAnArray(yi, yi_dst)
            accuracy += np.sum(np.argmax(yi, axis=1) == np.argmax(yi_dst, axis=1))
            total_elementcnt += yi.shape[0]
        return loss, accuracy / total_elementcnt

    def fit(self, x, y_dst):
        y = self.net.forward(*x)
        loss, accuracy = self.evaluate(y, y_dst)
        deriv_loss = [
            self.deriv_lossAnArray(yi, yi_dst) for yi, yi_dst in zip(y, y_dst)
        ]
        self.net.backward(*deriv_loss)
        return loss, accuracy


class CrossEntropy(Loss):
    def lossAnArray(self, y, y_dst):
        return -np.sum(np.where(y_dst == 1, np.log(y), 0))

    def deriv_lossAnArray(self, y, y_dst):
        y_clip = np.clip(y, cfg.divide_epsilon, 1 - cfg.divide_epsilon)
        return np.where(y_dst == 1, 1 / y_clip, 0)

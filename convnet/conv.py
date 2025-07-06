from types import NoneType
from .. import base, cfg, np, sci, misc
from .boost import conv2db


class Flip(base.Operation):
    def __init__(self, input: base.Operation, axis: tuple | NoneType = None):
        self.axis = axis
        super().__init__(input)

    def forwardUnwrap(self):
        self.output = np.flip(self.input.output, self.axis)

    def backwardUnwrap(self):
        self.input.grad += Flip(self.grad, self.axis)


def conv2dds(
    weight: base.Operation, kernels: base.Operation, input: base.Operation, padding=0
):
    assert (
        input.output.ndim == 3 and kernels.output.ndim == 3 and weight.output.ndim == 2
    )
    assert input.output.shape[0] == weight.output.shape[0]
    assert weight.output.shape[1] == kernels.output.shape[0]
    return conv2ddeepwise(weight, Conv2dSpare(kernels, input, padding))


def conv2ddeepwise(weight: base.Operation, input: base.Operation):
    assert input.output.ndim == 3 and weight.output.ndim == 2
    assert input.output.shape[0] == weight.output.shape[1]
    return misc.Einsum("oc,chw->ohw", weight, input)


class Conv2dSpare(base.Operation):
    def __init__(self, kernels: base.Operation, input: base.Operation, padding=0):
        self.kernels = kernels
        self.input = input
        self.padding = padding
        super().__init__(kernels, input)

    def forwardUnwrap(self):
        assert self.input.output.ndim == 3 and self.kernels.output.ndim == 3
        self.output = conv2db(self.input.output, self.kernels.output, self.padding)

    def backwardUnwrap(self):
        self.input.grad += conv2db(
            self.grad, Flip(self.kernels.output, (1, 2)), self.padding
        )

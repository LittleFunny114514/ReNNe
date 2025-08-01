from types import NoneType
from .. import base, np, misc
from .boost import conv2db
import re

class Flip(base.Operation):
    def __init__(self, input: base.Operation, axis: tuple | NoneType = None):
        self.axis = axis
        super().__init__(input)

    def forwardUnwrap(self):
        self.output = np.flip(self.inputs[0].output, self.axis)

    def backwardUnwrap(self):
        self.input.grad += Flip(self.grad, self.axis)


def conv2dds(
    weight: base.Operation, input: base.Operation, kernels: base.Operation, padding=0
):
    return conv2ddeepwise(weight, Conv2dSpare(input, kernels, padding))


def conv2ddeepwise(weight: base.Operation, input: base.Operation):
    return misc.Einsum("oc,chw->ohw", weight, input)


class Conv2dSpare(base.Operation):

    def __init__(
        self,
        input: base.Operation,
        kernels: base.Operation,
        padding: tuple[int] | base.Operation = (0, 0),
    ):
        """
        Argument padding:
            if padding is an int: padh = padw = padding
            if padding is a pair of ints: (padh, padw)
            if padding is an Operation: the shape of self.output is same with padding.output
        """
        self.kernels = kernels
        self.input = input
        self.padding = padding
        assert type(self.padding) in [int, tuple] or isinstance(
            self.padding, base.Operation
        )
        if isinstance(self.padding, tuple):
            assert len(self.padding) == 2
            assert type(self.padding[0]) == int
            assert type(self.padding[1]) == int

        super().__init__(self.input, self.kernels)

    def forwardUnwrap(self):
        padx = 0
        pady = 0
        if isinstance(self.padding, int):
            padx = pady = self.padding
        elif isinstance(self.padding, tuple):
            padx, pady = self.padding
        elif isinstance(self.padding, base.Operation):
            outrows_valid = (
                self.input.output.shape[1] - self.kernels.output.shape[1] + 1
            )
            outcols_valid = (
                self.input.output.shape[2] - self.kernels.output.shape[2] + 1
            )
            outrows_tgt = self.padding.output.shape[1]
            outcols_tgt = self.padding.output.shape[2]
            assert (outrows_tgt - outrows_valid) % 2 == 0
            assert (outcols_tgt - outcols_valid) % 2 == 0
            padx = (outcols_tgt - outcols_valid) // 2
            pady = (outrows_tgt - outrows_valid) // 2
        self.output = conv2db(
            self.input.output, self.kernels.output, padx=padx, pady=pady
        )

    def backwardUnwrap(self):
        self.input.grad += Conv2dSpare(
            self.grad, Flip(self.kernels, (1, 2)), self.input
        )
        self.kernels.grad += Conv2dSpare(self.input, self.grad, self.kernels)

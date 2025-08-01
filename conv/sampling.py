from typing import Callable, Literal
from .. import base, np, sci
from . import boost


class Pooling2D(base.Operation):

    def __init__(
        self,
        input: base.Operation,
        size: int,
        mode: Literal["avg", "max"] | Callable = "max",
    ):
        self.input: base.Operation = input
        self.size = size
        self.mode = mode
        assert mode in ["max", "avg"] or hasattr(
            mode, "__call__"
        ), "Unsupported pooling mode"
        super().__init__(input)

    def forwardUnwrap(self):

        if self.mode in ["max", "avg"]:
            self.output = boost.pooling2db(
                self.input.output, self.size,0, self.mode == "max"
            )
        else:
            self.output = self.mode(self.input.output, self.size)

    def backwardUnwrap(self):
        if self.mode == "max":
            self.input.grad += BwdMaxPooling2D(self.input, self.grad, self.size)
        elif self.mode == "avg":
            self.input.grad += NearestSampling2D(self.grad, self.size) * (self.size**-2)


class BwdMaxPooling2D(base.Operation):
    def __init__(self, input: base.Operation, outgrad: base.Operation, size: int):
        self.input: base.Operation = input
        self.outgrad: base.Operation = outgrad
        self.size = size
        super().__init__(input, outgrad)

    def forwardUnwrap(self):

        def f(x):
            return x if x != 0 else None

        outw = self.outgrad.output.shape[1]
        outh = self.outgrad.output.shape[2]
        inpw = self.input.output.shape[1]
        inph = self.input.output.shape[2]
        self.output = boost.bwdmaxpooling2db(
            self.input.output, self.outgrad.output, self.size
        )
        return
    def backwardUnwrap(self):
        raise NotImplementedError


class Roll(base.Operation):
    def __init__(self, input: base.Operation, shift: tuple[int]):
        self.input: base.Operation = input
        self.shift = shift
        super().__init__(input)

    def forwardUnwrap(self):
        assert (
            len(self.shift) == self.input.output.ndim
        ), "shift must be same length as input"
        self.output = np.roll(self.input.output, self.shift)

    def backwardUnwrap(self):
        self.input.grad += Roll(self.grad, tuple(map(lambda x: -x, self.shift)))


class Pad(base.Operation):
    def __init__(self, input: base.Operation, pad: tuple[tuple[int, int]]):
        self.input: base.Operation = input
        self.pad = pad
        self.maxpad0 = tuple(map(lambda x: (max(x[0], 0), max(x[1], 0)), pad))
        self.minpad0 = tuple(map(lambda x: (min(x[0], 0), min(x[1], 0)), pad))
        t = tuple(map(lambda x: (-x[0], x[1]), self.minpad0))
        self.slices = tuple(
            map(
                lambda x: slice(
                    None if x[0] == 0 else x[0], None if x[1] == 0 else x[1]
                ),
                t,
            )
        )
        super().__init__(input)

    def forwardUnwrap(self):
        tmp = np.pad(self.input.output, self.maxpad0)
        self.output = tmp[self.slices]

    def backwardUnwrap(self):
        self.input.grad += Pad(
            self.grad, tuple(map(lambda x: (-x[0], -x[1]), self.pad))
        )


class GrilleSampling2D(base.Operation):
    def __init__(
        self,
        input: base.Operation,
        size: int,
        mode: Literal["ups", "downs"] = "ups",
        shift: tuple[int, int] = (0, 0),
    ):
        self.input: base.Operation = input
        self.size = size
        self.mode = mode == "ups"
        self.shift = shift
        super().__init__(input)

    def forwardUnwrap(self):
        inp_shape = self.input.shape.output.shape
        if self.mode:
            self.output = np.zeros(
                (inp_shape[0], inp_shape[1] * self.size, inp_shape[2] * self.size)
            )
            self.output[:, :: self.size, :: self.size] = np.roll(
                self.input.output, self.shift, axis=(1, 2)
            )
        else:
            self.output = np.zeros(
                (inp_shape[0], inp_shape[1] // self.size, inp_shape[2] // self.size)
            )
            self.output = np.roll(self.input.output, self.shift, axis=(1, 2))[
                :, :: self.size, :: self.size
            ]

    def backwardUnwrap(self):
        opposite_mode = {"ups": "downs", "downs": "ups"}[self.mode]
        sampled_grad = GrilleSampling2D(self.grad, self.size, opposite_mode)
        rolled_grad = Roll(sampled_grad, (0, -self.shift[0], -self.shift[1]))
        self.input.grad += rolled_grad


class NearestSampling2D(base.Operation):
    def __init__(
        self,
        input: base.Operation,
        size: int,
    ):
        self.input: base.Operation = input
        self.size = size
        super().__init__(input)

    def forwardUnwrap(self):
        self.output = sci.ndimage.zoom(self.input.output, (1, self.size, self.size), 0)

    def backwardUnwrap(self):
        self.input.grad += Pooling2D(self.grad, self.size, "avg") * (self.size**2)


def bilinearInterpolation2d(x: base.Operation, scaling_ratio: tuple[int, int]):
    delta = scaling_ratio[1] / scaling_ratio[0]
    scale = scaling_ratio[0] / scaling_ratio[1]
    x_grill = np.zeros((scaling_ratio[1], scaling_ratio[1]), dtype=base.Operation)
    for i, j in np.ndindex(scaling_ratio[1], scaling_ratio[1]):
        x_grill[i, j] = GrilleSampling2D(x, scaling_ratio[1], "downs", (i, j))
    ret = base.input(0)
    for i, j in np.ndindex(scaling_ratio[0], scaling_ratio[0]):
        delta_x = np.mod(delta * i, scaling_ratio[1])
        delta_y = np.mod(delta * j, scaling_ratio[1])
        I = int(delta * i) // scaling_ratio[1]
        J = int(delta * j) // scaling_ratio[1]
        y_grill_ij = (
            (1 - delta_x) * (1 - delta_y) * x_grill[I, J]
            + (1 - delta_x) * delta_y * x_grill[I, J + 1]
            + delta_x * (1 - delta_y) * x_grill[I + 1, J]
            + delta_x * delta_y * x_grill[I + 1, J + 1]
        )
        ret += Roll(GrilleSampling2D(y_grill_ij, scaling_ratio[0], "ups"), (0, i, j))
    ret.inputs = ret.inputs[1:]
    return ret

from typing import Literal
from .. import base, cfg, np, misc, sci
from . import boost


class Pooling2D(base.Operation):
    def __init__(self, input, size: int, mode: Literal["avg", "max"] = "max"):
        self.input = input
        self.size = size
        self.mode = mode
        assert mode in ["max", "avg"], "Unsupported pooling mode"
        super().__init__(input)

    def forwardUnwrap(self):
        channels = self.input.shape[0]
        fmw = self.input.shape[2]
        fmh = self.input.shape[1]
        assert fmw % self.size == 0, "fmw must be divisible by size"
        assert fmh % self.size == 0, "fmh must be divisible by size"

        if self.mode == "max":
            self.output = sci.ndimage.maximum_filter(
                self.input.output, (1, self.size, self.size)
            )
        elif self.mode == "avg":
            self.output = sci.ndimage.uniform_filter(
                self.input.output, (1, self.size, self.size)
            )

    def backwardUnwrap(self):
        channels = self.input.shape[0]
        fmw = self.input.shape[2]
        fmh = self.input.shape[1]
        assert fmw % self.size == 0, "fmw must be divisible by size"
        assert fmh % self.size == 0, "fmh must be divisible by size"
        if self.mode == "max":
            self.grad += BwdMaxPooling2D(self.input, self.output, self.size)
        elif self.mode == "avg":
            self.grad += AdvSampling2D(self.input, self.size) * (1 / self.size**2)


class BwdMaxPooling2D(base.Operation):
    def __init__(self, input: base.Operation, outgrad: base.Operation, size: int):
        self.input: base.Operation = input
        self.outgrad: base.Operation = outgrad
        self.size = size
        super().__init__(input, outgrad)

    def forwardUnwrap(self):
        self.output = boost.bwdmaxpooling2db(self.input.output, self.outgrad, self.size)

    def backward(self):
        self.outgrad.grad += Pooling2D(self.input, self.size, "max") * self.outgrad


class AdvSampling2D(base.Operation):
    def __init__(
        self,
        input: base.Operation,
        scaling_ratio: tuple[int, int] = (2, 1),
        mode: Literal["nearest", "bilinear", "grille"] = "nearest",
    ):
        """
        size: (p,q): numerator and denominator of the upsampling ratio
        mode: "nearest", "bilinear", "grille"
        """
        assert mode in ["nearest", "bilinear", "grille"], "Unsupported upsampling mode"
        assert len(self.scaling_ratio) == 2, "scaling_ratio must be a tuple of length 2"
        assert (
            type(self.scaling_ratio[0]) == int and type(self.scaling_ratio[1]) == int
        ), "scaling_ratio must be a tuple of int"
        self.input: base.Operation = input
        self.scaling_ratio = scaling_ratio
        self.mode = mode
        if mode == "nearest":
            assert (
                self.scaling_ratio[1] == 1
            ), "scaling_ratio must be an integer if mode is nearest"
        elif mode == "grille":
            assert (
                self.scaling_ratio[0] == 1 or self.scaling_ratio[1] == 1
            ), "scaling_ratio must be an integer or the reciprocal of an integer if mode is grille"
        super().__init__(input)

    def forwardUnwrap(self):
        if self.mode == "nearest":
            self.output = sci.ndimage.zoom(
                self.input.output,
                (1, self.scaling_ratio[0], self.scaling_ratio[0]),
                order=0,
                mode="nearest",
            )
        elif self.mode == "bilinear":
            scale = self.scaling_ratio[0] / self.scaling_ratio[1]
            self.output = sci.ndimage.zoom(
                self.input.output, (1, scale, scale), order=1, mode="nearest"
            )
        elif self.mode == "grille":
            self.output = boost.grilleInterpolation2db(
                self.input.output, max(self.scaling_ratio), self.scaling_ratio[1] == 1
            )

    def backwardUnwrap(self):
        if self.mode == "nearest":
            self.grad += Pooling2D(self.input, self.scaling_ratio[0], "avg") * (
                self.scaling_ratio[0] ** 2
            )
        elif self.mode == "bilinear":
            scale = self.scaling_ratio[0] / self.scaling_ratio[1]
            delta = scale - self.scaling_ratio[1] / self.scaling_ratio[0]

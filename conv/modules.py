from typing import Callable, Literal
from .. import base, moduleblock, misc, cfg
from . import conv, sampling

np = cfg.np


class Conv2DParamInit(base.ParameterInitializer):
    def __init__(
        self,
        mode: Literal["normal", "uniform"] = "normal",
    ):
        assert mode in ["normal", "uniform"]
        self.mode = mode

    def __call__(self, value: np.ndarray, factor=1):
        if value.ndim == 4:  # std
            sqrdev_deno = (
                value.shape[0] + value.shape[1] * value.shape[2] * value.shape[3]
            )
            if self.mode == "normal":
                sqrdev = 2 / sqrdev_deno
                value[:] = np.random.normal(0, factor * sqrdev**0.5, value.shape)
            else:
                sqrdev = 6 / sqrdev_deno
                value[:] = np.random.uniform(
                    -factor * (sqrdev**0.5), factor * sqrdev**0.5, value.shape
                )
        elif value.ndim == 3:  # ds
            sqrdev_deno = value.shape[1] * value.shape[2] + 1
            if self.mode == "normal":
                sqrdev = 2 / sqrdev_deno
                value[:] = np.random.normal(0, factor * sqrdev**0.5, value.shape)
            else:
                sqrdev = 6 / sqrdev_deno
                value[:] = np.random.uniform(
                    -factor * (sqrdev**0.5), factor * sqrdev**0.5, value.shape
                )
        # elif param.ndim == 5:  # grp


class Conv2DDS(moduleblock.Module):
    def init(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation: Callable[[base.Operation], base.Operation] = lambda x: x * (x > 0),
        padding=0,
        bias=True,
    ):
        param_init = Conv2DParamInit()
        self.krnls = base.parameter((in_channels, kernel_size, kernel_size), param_init)
        self.ds_w = base.parameter((out_channels, in_channels))
        self.ds_b = base.parameter((out_channels, 1, 1)) if bias else None
        self.params = [self.krnls, self.ds_w]
        if self.ds_b is not None:
            self.params.append(self.ds_b)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation if activation is not None else lambda x: x
        self.io_shape = ((in_channels, -1, -1),), ((out_channels, -1, -1),)

    def forwardUnwrap(self, x: base.Operation) -> base.Operation:
        x = conv.conv2dds(self.ds_w, x, self.krnls, self.padding)
        if self.ds_b is not None:
            x += self.ds_b
        return self.activation(x)

    def info(self):
        return {"activation": self.activation.__name__}


class Pooling2D(moduleblock.Module):
    def init(self, size, mode: Literal["avg", "max"] = "max"):
        self.kernel_size = size
        self.mode = mode
        self.io_shape = ((-1, -1, -1), (-1, -1, -1))

    def forwardUnwrap(self, x: base.Operation) -> base.Operation:
        return sampling.Pooling2D(x, self.kernel_size, self.mode)

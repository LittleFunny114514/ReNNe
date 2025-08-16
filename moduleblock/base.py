from types import NoneType
from collections.abc import Iterable
from typing import Any, Callable, Dict, Protocol, Tuple

from .. import base, cfg
from ..misc import phase1 as misc

np = cfg.np


class Module:
    def __init__(self, *args, **kwargs):
        self.params: list[base.Data] = []
        self.sublayers: list[Module] = []
        self.io_shape = ((-1,),), ((-1,),)
        self.init(*args, **kwargs)

    def init(self, *args, **kwargs):
        pass

    def forwardUnwrap(
        self, *x: base.Operation
    ) -> tuple[base.Operation] | base.Operation:
        raise NotImplementedError

    def forward(
        self, *x: base.Operation, reserve_output: bool = False
    ) -> tuple[base.Operation]:
        y = self.forwardUnwrap(*x)
        if isinstance(y, base.Operation):
            y = (y,)
        assert isinstance(y, Iterable)
        y = tuple(y)
        if reserve_output:
            for yi in y:
                yi.output_reserved = True
        return y

    def backward(self):
        for param in self.params:
            param.backward()
        for sublayer in self.sublayers:
            sublayer.backward()

    def __call__(
        self, *x: np.ndarray, clear_operation_correlates=True
    ) -> tuple[np.ndarray]:
        if clear_operation_correlates:
            self.clearOperationCorrelates()
        xo = [base.input(xi) for xi in x]
        y = self.forward(*xo)
        for out in y:
            out.forward()
        return tuple([out.output for out in y])

    def setOptimizer(self, optimizer: base.Optimizer):
        for param in self.params:
            param.setOptimizer(optimizer)
        for sublayer in self.sublayers:
            sublayer.setOptimizer(optimizer)

    def calcGradients(self):
        for param in self.params:
            param.grad.forward()
        for sublayer in self.sublayers:
            sublayer.calcGradients()

    def clearOperationCorrelates(self):
        for param in self.params:
            param.grad = None
            param.fwdstate = False
            param.bwdvisited = False
            param.clearOutputs(True)
        for sublayer in self.sublayers:
            sublayer.clearOperationCorrelates()

    def save(self, path: str, key: str | NoneType = None):
        if key is None:
            np.savez_compressed(path, **self.save(path, ""))
            return
        key += self.__class__.__name__
        ret: dict = {
            f"{key}[{i}]": self.params[i].value for i in range(len(self.params))
        }

        for i, sublayer in enumerate(self.sublayers):
            ret.update(sublayer.save(path, f"{key}.no{i}_"))
        return ret

    def load(self, path: str, data=None, key: str = ""):
        if data is None:
            data = np.load(path)
        key += self.__class__.__name__
        for i, sublayer in enumerate(self.sublayers):
            sublayer.load(path, data, f"{key}.no{i}_")
        for i, param in enumerate(self.params):
            param.value = data[f"{key}[{i}]"]

    def update(self):
        for param in self.params:
            param.update()
        for sublayer in self.sublayers:
            sublayer.update()

    def info(self) -> dict:
        return dict()

    def toString(self, indent: int = 0) -> str:
        indent_str = "|" + " " * (cfg.INDENT_LEN - 1)
        last_indent = "|" + "-" * (cfg.INDENT_LEN - 1)
        ret = (
            indent_str * (indent - 1)
            + (last_indent if indent > 0 else "")
            + f"{self.__class__.__name__}("
            + repr(self.io_shape)
            + ", ".join([""] + [f"{key}={value}" for key, value in self.info().items()])
            + ")\n"
        )

        for sublayer in self.sublayers:
            ret += sublayer.toString(indent + 1)
        return ret

    def __repr__(self) -> str:
        return self.toString()


class Sequential(Module):
    def init(self, *modules: Module):
        self.sublayers = [*modules]

    def forwardUnwrap(self, *x: base.Operation) -> tuple[base.Operation]:
        for sublayer in self.sublayers:
            x = sublayer.forward(*x)
        return x

    @property
    def io_shape(self):
        ishape, _ = self.sublayers[0].io_shape
        _, oshape = self.sublayers[-1].io_shape
        return ishape, oshape

    @io_shape.setter
    def io_shape(self, value: tuple[tuple, tuple]):
        pass


class ResidualSequential(Module):

    @property
    def io_shape(self):
        return self.seq.io_shape

    @io_shape.setter
    def io_shape(self, value: tuple[tuple, tuple]):
        pass

    def init(self, *module: Module, postnorm: Module | NoneType = None):
        self.seq = Sequential(*module)
        self.postnorm = postnorm
        self.sublayers = [self.seq, self.postnorm]

    def forwardUnwrap(self, x: base.Operation) -> base.Operation:
        if self.postnorm is not None:
            return self.postnorm.forward(self.seq.forwardUnwrap(x) + x)
        else:
            return self.seq.forwardUnwrap(x) + x

    def info(self) -> dict:
        return {"norm": self.postnorm.__class__.__name__}


class Linear(Module):

    def init(
        self,
        in_features: int,
        out_features: int,
        enable_bias: bool = True,
        activation: Callable[[base.Operation], base.Operation] = misc.Tanh,
    ):
        self.weight = base.parameter((in_features, out_features))
        self.bias = base.parameter((out_features,)) if enable_bias else None
        self.params = [self.weight, self.bias] if enable_bias else [self.weight]
        self.activation = activation if activation is not None else lambda x: x
        self.io_shape = (
            (
                (
                    -1,
                    in_features,
                ),
            ),
            (
                (
                    -1,
                    out_features,
                ),
            ),
        )

    def forwardUnwrap(self, x: base.Operation) -> base.Operation:
        y = self.activation(
            x @ self.weight + self.bias if self.bias is not None else x @ self.weight
        )
        return y

    def info(self) -> dict:
        return {
            "activation": self.activation.__name__,
            "use_bias": self.bias is not None,
        }


class Reshape(Module):
    """
    This class would be removed in next alpha version
    """

    def init(self, *shape):
        self.shape = shape
        self.io_shape = (((0,),), (shape,))

    def forwardUnwrap(self, x: base.Operation) -> base.Operation:
        return x.reshape(*self.shape)

    def info(self) -> dict:
        return {"shape": self.shape}


class Lambda(Module):

    def init(
        self,
        func: Callable,
        io_shape: tuple[tuple, tuple] = ((-1,), (-1,)),
        **kwargs,
    ):
        self.func = func
        self.func_kwargs = kwargs
        self.io_shape = (io_shape[0],), (io_shape[1],)

    def forwardUnwrap(self, *x: base.Operation) -> tuple[base.Operation]:
        return self.func(*x, **self.func_kwargs)

    def info(self) -> dict:
        return {"func": self.func.__name__}

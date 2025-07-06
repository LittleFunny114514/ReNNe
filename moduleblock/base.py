from types import NoneType
from typing import override
from collections.abc import Iterable
import gc

from .. import base, cfg
from ..misc import phase1 as misc

np = cfg.np


class Module:
    def __init__(self, *args, **kwargs):
        self.params: list[base.Data] = []
        self.sublayers: list[Module] = []
        self.init(*args, **kwargs)

    def getIOShape(self):
        """
        return: ((input_shape1,output_shape1),...)
        zero if the sidelen of this axis is arbitary.
        """
        raise NotImplementedError

    def init(self, *args, **kwargs):
        pass

    def forwardUnwrap(self, *x: base.Operation) -> tuple[base.Operation]:
        raise NotImplementedError

    def forward(
        self, *x: base.Operation, reserve_output: bool = False
    ) -> tuple[base.Operation]:
        y = self.forwardUnwrap(*x)
        if isinstance(y, base.Operation):
            return (y,)
        assert isinstance(y, Iterable)
        if not isinstance(y, tuple):
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
            param.fwdvisited = False
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
            + repr(self.getIOShape())
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

    def forwardUnwrap(self, *x: base.Operation) -> base.Operation:
        for sublayer in self.sublayers:
            x = sublayer.forward(*x)
        return x

    def getIOShape(self):
        ishape, _ = self.sublayers[0].getIOShape()
        _, oshape = self.sublayers[-1].getIOShape()
        return ishape, oshape


class ResidualSequential(Module):
    def getIOShape(self):
        return self.seq.getIOShape()

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
        activation=misc.Tanh,
    ):
        self.weight = base.parameter((in_features, out_features))
        self.bias = base.parameter((out_features,)) if enable_bias else None
        self.params = [self.weight, self.bias] if enable_bias else [self.weight]
        self.activation = activation if activation is not None else lambda x: x

    def getIOShape(self):
        return ((0, self.weight.value.shape[0]),), ((0, self.weight.value.shape[1]),)

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
    def init(self, *shape):
        self.shape = shape

    def forwardUnwrap(self, x: base.Operation) -> base.Operation:
        return x.reshape(*self.shape)

    def getIOShape(self):
        return ((0,),), ((0,),)
    def info(self) -> dict:
        return {"shape": self.shape}
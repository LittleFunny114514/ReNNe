from collections import deque
from math import sqrt
from queue import Queue
from types import NoneType

import inspect
import asyncio
from typing import Iterable, Literal
from . import cfg

np = cfg.np
sci = cfg.sci
mp = cfg.mp


def getMalMulShape(shape0, shape1):
    if len(shape0) == 1 and len(shape1) == 1:
        assert shape0[0] == shape1[0]
        return (1,)
    elif len(shape0) == 1:
        assert shape0[0] == shape1[0]
        return (shape1[1],)
    elif len(shape1) == 1:
        assert shape0[1] == shape1[0]
        return (shape0[0],)
    else:
        assert shape0[1] == shape1[0]
        return (shape0[0], shape1[1])


class ParameterInitializer:

    def __init__(self, type: Literal["normal", "uniform"]):
        assert type in ["normal", "uniform"]
        self.type = type

    def __call__(self, value: np.ndarray, sqrdev=0, factor=1):
        assert isinstance(value, np.ndarray)
        sum = 0
        for num in value.shape:
            sum += num

        if self.type == "normal":
            stddev = sqrt(2 / sum) if sqrdev == 0 else sqrdev
            value[:] = np.random.normal(0, stddev * factor, value.shape).astype(
                cfg.dtype
            )
        elif self.type == "uniform":
            stddev = sqrt(6 / sum) if sqrdev == 0 else sqrdev
            value[:] = np.random.uniform(
                -stddev * factor, stddev * factor, value.shape
            ).astype(cfg.dtype)


class Optimizer:
    """
    Reverse gradient descent optimizer. \n
    Note that all optimizers based on this are reverse, i.e. gradient rising. \n
    Support multiple levels. \n
    """

    def __init__(self, lr, nthgrad=(1,)):
        self.lr = lr
        self.nthgrad = nthgrad

    def createLike(self):
        return Optimizer(self.lr, self.nthgrad)

    def setNthGrad(self, nthgrad: tuple):
        self.nthgrad = nthgrad
        return self

    def setParameter(self, parameter):
        self.parameter = parameter
        return self

    def update(self, grad: np.ndarray):
        self.parameter.grad += self.lr * grad

    def __call__(self):
        grad = cfg.dtype(0)
        for nth in self.nthgrad:
            grad += eval("self.parameter" + nth * ".grad" + ".output")
        self.update(grad)


def JsonFormat(obj, indent=0) -> str:
    # Teanslate a list or a dict to a multiline and indented json string
    indent_str = " " * cfg.INDENT_LEN * indent
    if isinstance(obj, dict):
        ret = "{\n"
        for key, value in obj.items():
            ret += (
                indent_str
                + f"    {JsonFormat(key, indent + 1)}: {JsonFormat(value, indent + 1)},\n"
            )
        return ret[:-2] + "\n" + indent_str + "}"
    elif isinstance(obj, list):
        ret = "[\n"
        for item in obj:
            ret += indent_str + "    " + JsonFormat(item, indent + 1) + ",\n"
        return ret[:-2] + "\n" + indent_str + "]"
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    else:
        return repr(obj)


class OperationDbgInfo:
    def __init__(self):
        self.callstackoncreate = []
        self.fatherop: OperationDbgInfo = None

    def __repr__(self):
        return f"OperationDbgInfo({JsonFormat(self.callstackoncreate)},\n fatherop = {self.fatherop})"


class OperatingError(Exception):
    def __init__(self, message, dbginfo: OperationDbgInfo, exception: Exception = None):
        self.message = message
        self.dbginfo: OperationDbgInfo = dbginfo
        self.original_exception = exception

    def __repr__(self):
        if self.original_exception:
            return f"{self.message}:\n Operation Created at {self.dbginfo}\n {str(self.original_exception)}"
        return super().__str__()

    def __str__(self):
        if self.original_exception:
            return f"{self.message}:\n Operation Created at {JsonFormat(self.dbginfo)}\n {str(self.original_exception)}"
        return super().__str__()


def NodeRecursion(state_name: str = ""):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.__dict__[state_name] = False
                raise OperatingError(
                    f"Error during operation {self.__class__.__name__}.{func.__name__}",
                    self.dbginfo,
                    e,
                ) from e

        async def async_wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                self.__dict__[state_name] = False
                raise OperatingError(
                    f"Error during operation {self.__class__.__name__}.{func.__name__}",
                    self.dbginfo,
                    e,
                ) from e

        if cfg.CREATE_DBGINFO:
            return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
        else:
            return func

    return decorator


class AsyncIteratable:
    itab: Iterable

    def __init__(self, itab: Iterable):
        self.itab = itab

    def __getitem__(self, idx):
        return self.itab[idx]

    def __aiter__(self):
        self.__idx = 0
        return self

    async def __anext__(self):
        if self.__idx >= len(self.itab):
            raise StopAsyncIteration
        value = self.itab[self.__idx]
        self.__idx += 1
        return value


class Operation:
    """
    The Operation class is the base class for all operations in neural networks. \n\n

    Attribute: \n
    inputs, outputs:  List of input/output operations for this operation\n
    output:  The output value of this operation is of type numpy scalar or tensor\n
    grad:  The gradient of this operation is of type Operation\n
    fwdvisited:  Has forward propagation accessed the flag\n
    bwdvisited:  Has backpropagation accessed the flag\n\n

    method:\n
    init(*input):  Initialization method, can be overridden by subclasses, sometimes it is also necessary to directly override __init__\n
    forwardUnwrap():  The specific calculation of forward propagation requires subclass implementation\n
    backwardUnwrap():  The specific calculation of backpropagation requires subclass implementation\n
    forward():  Perform forward propagation\n
    backward():  Perform backpropagation\n
    reset(type='both'):  Reset calculation status\n
    setvalue(value):  Set input value (only available in subclass Parameter)\n
    update():  Update gradient (only valid in subclass Parameter)\n
    __repr_(): Returns a string representation\n
    __str__(): Returns a string representation\n\n

    Supported Operators:\n
    +, -, *, /, @ (matrix multiplication), -, .t() (transpose), .reshape() (shape transforms)
    """

    def __init__(self, *input):
        if cfg.CREATE_DBGINFO:
            self.dbginfo: OperationDbgInfo = OperationDbgInfo()
            self.dbginfo.callstackoncreate = [
                {
                    "filepath": frame.filename,
                    "lineno": frame.lineno,
                    "function": frame.function,
                    "code_context": frame.code_context,
                }
                for frame in inspect.stack()
                if "ReNNe" in frame.filename
            ]
        for inp in input:
            assert isinstance(inp, Operation)
        self.inputs: list[Operation] = [*input]  # src of this operate
        self.outputs: list[Operation] = []  # dst of this operate
        for inp in self.inputs:
            inp.outputs.append(self)
        self.output: np.ndarray | cfg.dtype = cfg.dtype(
            0
        )  # output value of this operation
        self.init(*input)
        self.grad: Operation | NoneType = None
        # ↑gradients of outputs of this operation
        self.fwdstate: bool = False  # DFS visited
        self.bwdvisited: bool = False
        self.deg = None
        self.outputs_length: NoneType | int = None
        self.output_reserved: bool = False

    def init(self, *input):
        pass

    def forwardUnwrap(self):
        # compute the output value based on input values
        raise NotImplementedError

    def backwardUnwrap(self):
        # compute the derivative of this operation w.r.t. it's inputs
        raise NotImplementedError

    @NodeRecursion("fwdstate")
    def forwardRecursively(self):
        # compute the output value based on input values
        if self.fwdstate == True:
            return
        # recursively compute the output value of all input operations

        for inp in self.inputs:
            inp.forwardRecursively()
        self.forwardUnwrap()

        if cfg.IMMEDIATE_REMOVE_HIDDEN:
            for inp in self.inputs:
                if (not inp.output_reserved) and np.array(
                    [oi.fwdstate for oi in inp.outputs]
                ).all():
                    inp.output = cfg.dtype()
                    inp.fwdstate = False

        self.fwdstate = True

    def forwardByTopsort(self):
        sorting: Queue[Operation] = Queue()

        def calc_deg(obj: Operation):
            if obj.deg is not None or obj.fwdstate:
                return
            for inp in obj.inputs:
                calc_deg(inp)
            obj.deg = [
                sum(1 for inp in obj.inputs if not inp.fwdstate),
                len(obj.outputs),
            ]

            if obj.deg[0] == 0 and not obj.fwdstate:
                sorting.put_nowait(obj)

        self.output_reserved = True
        calc_deg(self)
        while not sorting.empty():
            node: Operation = sorting.get_nowait()
            if not node.fwdstate:
                node.forwardUnwrap()
                node.fwdstate = True
            for inp in node.inputs:
                inp.deg[1] -= 1
                if inp.deg[1] == 0:
                    if cfg.IMMEDIATE_REMOVE_HIDDEN and not inp.output_reserved:
                        inp.output = cfg.dtype(0)
                        inp.fwdstate = False
                    inp.deg = None
            for out in node.outputs:
                if out.deg is not None:
                    out.deg[0] -= 1
                    if out.deg[0] == 0:
                        sorting.put_nowait(out)
            if node.deg[1] == 0:
                if cfg.IMMEDIATE_REMOVE_HIDDEN and not node.output_reserved:
                    node.output = cfg.dtype(0)
                    node.fwdstate = False
                node.deg = None

    def forward(self):
        self.forwardByTopsort()

    @NodeRecursion("bwdvisited")
    def backward(self):
        # compute the derivative of this operation w.r.t. it's inputs
        assert not cfg.NO_GRAD
        if self.bwdvisited:
            return

        def rec(obj: Operation):
            if obj.outputs_length is None:
                obj.outputs_length = len(obj.outputs)
                for out in obj.outputs:
                    rec(out)
                for inp in obj.inputs:
                    rec(inp)

        rec(self)
        # recursively create backward operations for all output operations
        if not (cfg.NO_GRAD or isinstance(self.grad, Operation)):
            self.grad = input(np.zeros_like(self.output))
        for i in range(self.outputs_length):
            self.outputs[i].backward()

        for inp in self.inputs:
            if inp.grad is None:
                inp.grad = input(np.zeros_like(inp.output))
        if cfg.CREATE_DBGINFO:
            self.grad.dbginfo.father = self.dbginfo
        if (
            isinstance(self.grad, Add)
            and isinstance(self.grad.inputs[0], input)
            and (self.grad.inputs[0].output == 0).all()
        ):
            self.grad.inputs = self.grad.inputs[1:]
            self.grad.setBoardCast(self)

        self.backwardUnwrap()
        self.bwdvisited = True

    @NodeRecursion()
    def reset(
        self, type: Literal["both", "fwd", "bwd"] = "both", zerooutput: bool = False
    ):
        assert type in ["both", "fwd", "bwd"]
        fwdv = False
        bwdv = False
        if (self.fwdstate or self.deg is not None) and type in ["fwd", "both"]:
            fwdv = True
            self.fwdstate = False
            self.deg = None
            if zerooutput:
                self.output = cfg.dtype(0)
        if self.bwdvisited and type in ["bwd", "both"]:
            bwdv = True
            self.bwdvisited = False
            self.grad = input(0)
            self.outputs = self.outputs[: self.outputs_length]
        if fwdv:
            for inp in self.inputs:
                inp.reset(type)
        if bwdv:
            for oup in self.outputs:
                oup.reset(type)

    def clearOutputs(self, recursive=False):
        stack = deque([self])
        cleared = set()
        while len(stack):
            node: Operation = stack.pop()
            cleared.add(node)
            node.output = cfg.dtype(0)
            for nod in node.inputs + node.outputs:
                if len(nod.outputs) and nod not in cleared:
                    stack.append(nod)
            node.outputs = []
            node.outputs_length = None
        """
        if self.outputs == []:
            return
        self_outputs = self.outputs
        self.output = cfg.dtype(0)
        self.outputs = []
        self.outputs_length = None

        if recursive:
            for out in self_outputs:
                out.clearOutputs(True)
        if recursive:
            for inp in self.inputs:
                inp.clearOutputs(True)
        """

    def update(self):
        assert not cfg.NO_GRAD
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str,self.inputs))})"

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Mul2(self, other)

    def __rmul__(self, other):
        return Mul2(self, other)

    def __truediv__(self, other):
        return DivElementWise(self, other)

    def __matmul__(self, other):
        return Matmul(self, other)

    def reshape(self, shape):
        if isinstance(shape, Operation):
            return ReshapeLike(self, shape)
        else:
            return Reshape(self, shape)

    def __sub__(self, other):
        return Add(self, other * -1)

    def __neg__(self):
        return Mul2(self, input(cfg.dtype(-1)))

    def __ge__(self, other):
        return Compare(self, other, "ge")

    def __gt__(self, other):
        return Compare(self, other, "g")

    def __le__(self, other):
        return Compare(self, other, "le")

    def __lt__(self, other):
        return Compare(self, other, "l")

    def t(self, axis: tuple = None):
        return Transpose(self, axis)


class Net:
    def __init__(self):
        self.parameters: list[parameter] = []

    def registerParameter(self, parameter):
        assert isinstance(parameter, parameter)
        self.parameters.append(parameter)
        return parameter

    def setOptimizer(self, optimizer: Optimizer):
        for param in self.parameters:
            param.setOptimizer(optimizer)

    def initParameters(self, parameter_initializer=ParameterInitializer("normal")):
        for param in self.parameters:
            parameter_initializer(param.value)

    def save(self, path):
        np.savez(
            path,
            **{str(i): self.parameters[i].value for i in range(len(self.parameters))},
        )

    def load(self, path):
        npzfile = np.load(path)
        for i in range(len(self.parameters)):
            self.parameters[i].value = npzfile[str(i)]

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *grads):
        raise NotImplementedError

    def update(self):
        for param in self.parameters:
            param.update()

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def __repr__(self):
        return f"Net({self.inputs},{self.parameters},{self.outputs})"

    def __str__(self):
        return self.__repr__()


class Data(Operation):
    def __init__(self, value: np.ndarray, trainable: bool = False):
        super().__init__()
        self.trainable = trainable
        self.value: np.ndarray = cfg.dtype(value)
        if self.trainable:
            self.optimizer: Optimizer = Optimizer(0.001)
            self.optimizer.setParameter(self)

    def setOptimizer(self, optimizer: Optimizer):
        assert self.trainable
        self.optimizer = optimizer.createLike()
        self.optimizer.setParameter(self)

    def forwardUnwrap(self):
        self.output = self.value

    def backwardUnwrap(self):
        if not (cfg.NO_GRAD or isinstance(self.grad, Operation)):
            self.grad = input(np.zeros_like(self.output))

    def update(self):
        if self.trainable:
            self.optimizer()
            self.grad.reset("fwd")

    def setvalue(self, value):
        self.value = cfg.dtype(value)

    def __repr__(self):
        return f"Data({self.trainable},{self.value.shape})"

    def __str__(self):
        return self.__repr__()


def parameter(shape, parameter_initializer=ParameterInitializer("normal")):
    param = Data(np.zeros(shape), True)
    parameter_initializer(param.value)
    return param


input = Data


class Operator(Operation):
    def __init__(self, *inputs):
        super().__init__(
            *[inp if isinstance(inp, Operation) else input(inp) for inp in inputs]
        )


class Add(Operator):
    def init(self, *inputs):
        self.output = np.zeros_like(self.inputs[0].output)
        self.boardcastwith: Operation = None

    def setBoardCast(self, other: Operation):
        self.boardcastwith = other
        return self

    def forwardUnwrap(self):
        if self.boardcastwith is None:
            self.boardcastwith = self.inputs[0]
        self.output = np.zeros_like(self.boardcastwith.output)
        dimhide = {i for i, l in enumerate(self.boardcastwith.output.shape) if l == 1}
        for inp in self.inputs:
            dimdiff = self.boardcastwith.output.ndim - inp.output.ndim
            if inp.output.shape == self.output.shape or (
                dimdiff >= 0
                and (
                    np.array(
                        [
                            self.output.shape[dimdiff + i] - inp.output.shape[i]
                            for i in range(inp.output.ndim)
                        ]
                    )
                    >= 0
                ).all()
            ):
                self.output += inp.output
                continue
            if dimdiff < 0:
                squeezed = np.sum(inp.output, axis=tuple(range(-dimdiff)))
            else:
                squeezed = inp.output
            # dimhide is the dims that should be summed int
            squeezed = np.sum(
                squeezed,
                axis=tuple(
                    (i - dimdiff for i in dimhide if i >= dimdiff)
                    if dimdiff >= 0
                    else dimhide
                ),
                keepdims=True,
            )
            self.output += squeezed

    def backwardUnwrap(self):
        for inp in self.inputs:
            inp.grad += self.grad

    def __add__(self, other):
        return Add(*self.inputs, other).setBoardCast(self.boardcastwith)

    def __radd__(self, other):
        return Add(other, *self.inputs).setBoardCast(self.boardcastwith)

    def __iadd__(self, other):
        return Add(*self.inputs, other).setBoardCast(self.boardcastwith)


class Sum(Operation):
    def __init__(self, input, axis=None, keepdims=False):
        super().__init__(input)
        self.axis = axis
        self.keepdims = keepdims

    def forwardUnwrap(self):
        self.output = np.sum(
            self.inputs[0].output, axis=self.axis, keepdims=self.keepdims
        )

    def backwardUnwrap(self):
        self.inputs[0].grad += JoinDim(self.grad, self.axis)


class Mul2(Operator):
    def init(self, a, b):
        assert (
            a.output.shape == b.output.shape or a.output.ndim == 0 or b.output.ndim == 0
        )

    def forwardUnwrap(self):
        self.output = self.inputs[0].output * self.inputs[1].output

    def backwardUnwrap(self):
        self.inputs[0].grad += self.grad * self.inputs[1]
        self.inputs[1].grad += self.grad * self.inputs[0]


class DivElementWise(Operator):
    def init(self, a, b):
        pass

    def forwardUnwrap(self):
        assert self.inputs[0].output.shape == self.inputs[1].output.shape
        self.output = self.inputs[0].output / self.inputs[1].output

    def backwardUnwrap(self):
        self.inputs[0].grad += self.grad / self.inputs[1]
        self.inputs[1].grad -= (
            self.grad * self.inputs[0] / (self.inputs[1] * self.inputs[1])
        )


class Matmul(Operation):

    def __init__(self, left, right, laxis=None, raxis=None, oaxis=None):
        super().__init__(left, right)
        self.laxis = laxis
        self.raxis = raxis
        self.oaxis = oaxis
        self.left = left
        self.right = right
        if self.raxis is None:
            self.raxis = self.laxis
        if self.laxis is not None:
            assert len(laxis) == len(raxis) == 2
        if self.oaxis is None and self.laxis == self.raxis:
            self.oaxis = self.laxis

    def forwardUnwrap(self):
        a1_shape = self.inputs[0].output.shape
        a2_shape = self.inputs[1].output.shape
        a1_ndim = len(a1_shape)
        a2_ndim = len(a2_shape)

        if self.laxis is not None:
            assert max(self.laxis) < a1_ndim
            assert max(self.raxis) < a2_ndim
            assert min(self.laxis) >= -a1_ndim
            assert min(self.raxis) >= -a2_ndim
            laxis = tuple(range(a1_ndim))
            laxis[-2], laxis[self.laxis[0]] = laxis[self.laxis[0]], laxis[-2]
            laxis[-1], laxis[self.laxis[1]] = laxis[self.laxis[1]], laxis[-1]
            raxis = tuple(range(a2_ndim))
            raxis[-2], raxis[self.raxis[0]] = raxis[self.raxis[0]], raxis[-2]
            raxis[-1], raxis[self.raxis[1]] = raxis[self.raxis[1]], raxis[-1]
            left = np.transpose(self.inputs[0].output, axes=laxis)
            right = np.transpose(self.inputs[0].output, axes=raxis)
            mul = np.matmul(left, right)
            if self.oaxis is not None:
                oaxis = tuple(range(max(a1_ndim, a2_ndim)))
                oaxis[-2], oaxis[self.oaxis[0]] = oaxis[self.oaxis[0]], oaxis[-2]
                oaxis[-1], oaxis[self.oaxis[1]] = oaxis[self.oaxis[1]], oaxis[-1]
                mul = np.transpose(mul, axes=oaxis)
            self.output = mul
            return

        if a1_ndim == 0 or a2_ndim == 0:
            self.output = self.inputs[0].output * self.inputs[1].output
        elif not (a1_ndim <= 1 and a2_ndim <= 1):
            self.output = np.matmul(self.inputs[0].output, self.inputs[1].output)
        elif a1_ndim == a2_ndim == 1:
            self.output = np.outer(self.inputs[0].output, self.inputs[1].output)
        else:
            raise NotImplementedError

    def backwardUnwrap(self):
        self.inputs[0].grad += Matmul(self.grad, self.inputs[1].t())
        self.inputs[1].grad += Matmul(self.inputs[0].t(), self.grad)


class ElementWise(Operation):
    def __init__(self, funcs, arr):
        self.funcs = funcs  # nth derivative (including 0th derivative)
        super().__init__(arr)

    def forwardUnwrap(self):
        self.output = self.funcs[0](self.inputs[0].output)

    def backwardUnwrap(self):
        self.inputs[0].grad += ElementWise(self.funcs[1:], self.inputs[0]) * self.grad


class Reshape(Operation):
    def __init__(self, arr, shape: tuple):
        self.shape = shape
        super().__init__(arr)

    def forwardUnwrap(self):
        self.output = self.inputs[0].output.reshape(self.shape)

    def backwardUnwrap(self):
        self.inputs[0].grad += self.grad.reshape(self.inputs[0])


class ReshapeLike(Operation):
    def __init__(self, arr, like):
        self.like = like
        super().__init__(arr, like)

    def forwardUnwrap(self):
        self.output = self.inputs[0].output.reshape(self.like.output.shape)

    def backwardUnwrap(self):
        self.inputs[0].grad += self.grad.reshape(self.inputs[0])


class Transpose(Operation):
    def __init__(self, arr, axis: tuple = None):
        self.axis = axis
        super().__init__(arr)

    def forwardUnwrap(self):
        if self.axis is None:
            self.output = self.inputs[0].output.transpose()
        else:
            self.output = self.inputs[0].output.transpose(self.axis)

    def backwardUnwrap(self):
        if self.axis is None:
            self.inputs[0].grad += Transpose(self.grad)
        else:
            axisinv = [0] * len(self.axis)
            for i in range(len(self.axis)):
                axisinv[self.axis[i]] = i
            self.inputs[0].grad += Transpose(self.grad, tuple(axisinv))


class JoinDim(Operation):
    def __init__(self, arr, dims):
        self.dims = dims
        super().__init__(arr)

    def forwardUnwrap(self):
        newshape = [1] * (len(self.dims) + self.inputs[0].output.ndim)
        i = 0
        for n in self.inputs[0].output.shape:
            while i in self.dims:
                newshape[i] = 1
                i += 1
            newshape[i] = n
            i += 1
        self.output = self.inputs[0].output.reshape(tuple(newshape))

    def backwardUnwrap(self):
        self.inputs[0].grad += Sum(self.grad, self.dims)


class Compare(Operator):
    def __init__(
        self, left, right, mode: Literal["l", "g", "le", "ge", "e", "ne"] = "g"
    ):
        assert mode in ["l", "g", "le", "ge", "e", "ne"]
        self.mode = mode
        self.left = left if isinstance(left, Operation) else input(left)
        self.right = right if isinstance(right, Operation) else input(right)

        super().__init__(left, right)

    def forwardUnwrap(self):
        if self.mode == "l":
            self.output = self.left.output < self.right.output
        elif self.mode == "g":
            self.output = self.left.output > self.right.output
        elif self.mode == "le":
            self.output = self.left.output <= self.right.output
        elif self.mode == "ge":
            self.output = self.left.output >= self.right.output
        elif self.mode == "e":
            self.output = self.left.output == self.right.output
        elif self.mode == "ne":
            self.output = self.left.output != self.right.output

    def backwardUnwrap(self):
        pass

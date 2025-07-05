from typing import Iterable, override
from .. import base, cfg

np = base.np


class Exp(base.Operation):
    def __init__(self, input):
        super().__init__(input)

    def forwardUnwrap(self):
        self.output = np.exp(self.inputs[0].output)

    def backwardUnwrap(self):
        self.inputs[0].grad += self * self.grad


class Ln(base.Operation):
    """
    logarithmic function with base e\n
    it isn't layer normalization
    """

    def __init__(self, input):
        super().__init__(input)

    def forwardUnwrap(self):
        self.output = np.log(self.inputs[0].output)

    def backwardUnwrap(self):
        self.inputs[0].grad += self.grad / self.inputs[0]


class Where(base.Operation):
    def __init__(self, input, condition, true_value, false_value):
        self.condition = condition
        self.input = input
        self.true_value = true_value
        self.false_value = false_value
        super().__init__(input, true_value, false_value)

    def forwardUnwrap(self):
        self.output = np.where(
            self.condition(self.inputs[0].output), self.true_value, self.false_value
        )

    def backwardUnwrap(self):
        self.true_value.grad += Where(self.condition, self.grad, 0)
        self.false_value.grad += Where(self.condition, 0, self.grad)


class Clip(base.Operation):
    def __init__(self, input, min, max):
        self.min = min
        self.max = max
        super().__init__(input)

    def forwardUnwrap(self):
        self.output = np.clip(self.inputs[0].output, self.min, self.max)

    def backwardUnwrap(self):
        self.inputs[0].grad += Where()


def sigmoid(x: base.Operation):
    return 1 / (Exp(-x) + 1)


def relu(x):
    return Where(x, lambda x: x >= 0, x, 0)


def swish(x):
    """
    x*sigmoid(x)
    """
    return x * sigmoid(x)


class Tanh(base.Operation):
    def forwardUnwrap(self):
        self.output = np.tanh(self.inputs[0].output)

    def backwardUnwrap(self):
        self.inputs[0].grad += self.grad * (-self * self + 1)


class Softmax(base.Operation):
    """
    softmax each row
    """

    def init(self, input):
        pass

    def forwardUnwrap(self):
        assert self.inputs[0].output.ndim in [1, 2]
        if self.inputs[0].output.ndim == 1:
            exp = np.exp(self.inputs[0].output - np.max(self.inputs[0].output))
            self.output = exp / np.sum(exp)
        else:
            exp = np.exp(
                self.inputs[0].output
                - np.max(self.inputs[0].output, axis=1, keepdims=True)
            )
            self.output = exp / np.sum(exp, axis=1, keepdims=True)

    def backwardUnwrap(self):
        if self.inputs[0].output.ndim == 1:
            self.inputs[0].grad += (self.grad - base.Matmul(self.grad, self)) * self
        else:
            self.inputs[0].grad += (
                self.grad - base.JoinDim(Einsum("bi,bi->b", self.grad, self), (1,))
            ) * self


class DiscreateDerivative:
    def __init__(self, func, dx=0.0001):
        self.func = func
        self.dx = dx

    def __getitem__(self, n: int):
        if isinstance(n, slice):
            return [
                self.__getitem__(i)
                for i in range(n.start, n.stop, 1 if n.step is None else n.step)
            ]
        assert n >= 0 and isinstance(n, int)
        n += 1

        def nthderiv(x):
            lst = [0] * n
            for i in range(n):
                lst[i] = self.func(x + i * self.dx)
            for i in range(n):
                for j in range(n - i - 1):
                    lst[j] = (lst[j + 1] - lst[j]) / self.dx
            return lst[0]

        return nthderiv


class Einsum(base.Operation):
    def __init__(self, equation, *inputs):
        self.equation = equation
        self.left, self.right = equation.split("->")
        self.left = self.left.split(",")
        super().__init__(*inputs)

    def forwardUnwrap(self):
        self.output = np.einsum(self.equation, *[inp.output for inp in self.inputs])

    def backwardUnwrap(self):
        for i in range(len(self.inputs)):
            eq = (
                ",".join(self.left[:i] + [self.right] + self.left[i + 1 :])
                + f"->{self.left[i]}"
            )
            self.inputs[i].grad += Einsum(eq, self.grad)


class Take(base.Operation):
    def __init__(self, input, access_expression):
        self.access_expression = access_expression
        super().__init__(input)

    def forwardUnwrap(self):
        self.output = self.inputs[0].output[
            self.access_expression
            + (slice(None),)
            * (self.inputs[0].output.ndim - len(self.access_expression))
        ]

    def backwardUnwrap(self):
        self.inputs[0].grad[
            self.access_expression
            + (slice(None),)
            * (self.inputs[0].output.ndim - len(self.access_expression))
        ] += self.grad


class WeightDecay(base.Optimizer):
    def __init__(self, decay_rate, suboptimizer):
        self.decay_rate = decay_rate
        self.suboptimizer = suboptimizer

    def setParameter(self, parameter):
        self.suboptimizer.setParameter(parameter)

    def createLike(self):
        return WeightDecay(self.decay_rate, self.suboptimizer.createLike())

    @property
    def lr(self):
        return self.suboptimizer.lr

    @lr.setter
    def lr(self, lr):
        self.suboptimizer.lr = lr

    def setParameter(self, parameter):
        return self.suboptimizer.setParameter(parameter)

    def __call__(self):
        self.suboptimizer.parameter.value *= 1 - self.suboptimizer.lr * self.decay_rate
        self.suboptimizer()


class Adam(base.Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0
        self.v = 0
        self.expbeta1 = 1
        self.expbeta2 = 1
        super().__init__(lr)

    def createLike(self):
        return Adam(self.lr, self.beta1, self.beta2)

    def setParameter(self, parameter):
        self.parameter = parameter
        self.m = np.zeros_like(parameter.value)
        self.v = np.zeros_like(parameter.value)

    def update(self, grad: np.ndarray):
        self.expbeta1 *= self.beta1
        self.expbeta2 *= self.beta2

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        self.parameter.value += (
            self.lr
            / (1 - self.expbeta1)
            * self.m
            / (np.sqrt(self.v / (1 - self.expbeta2)) + cfg.devide_epsilon)
        )


def adamW(lr, beta1=0.8, beta2=0.95, weight_decay=0.1):
    return WeightDecay(weight_decay, Adam(lr, beta1, beta2))


class RMSProp(base.Optimizer):
    def __init__(self, lr, beta=0.9):
        self.beta = beta
        super().__init__(lr)

    def createLike(self):
        return RMSProp(self.lr, self.beta)

    def setParameter(self, parameter):
        self.parameter = parameter
        self.v = np.zeros_like(parameter.value)

    def update(self, grad: np.ndarray):
        self.v = self.beta * self.v + (1 - self.beta) * grad**2
        self.parameter.value += self.lr * grad / (np.sqrt(self.v) + cfg.divide_epsilon)

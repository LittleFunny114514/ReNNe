from .. import cfg, base

np = cfg.np


class NetStatic(base.Net):
    def __init__(self):
        super().__init__()
        self.inputs: list[base.Operation] = []
        self.outputs: list[base.Operation] = []

    def register(self, type: str, *operations):
        assert type in ["input", "parameter", "output"]
        assert np.array([isinstance(op, base.Operation) for op in operations]).all()
        for op in operations:
            eval("self." + type + "s.append(op)")

    def build(self):
        for op in self.outputs:
            op.grad = base.input(np.zeros_like(op.output))
            op.forward()
        for op in self.inputs:
            op.backward()
        for op in self.parameters:
            op.backward()
        pass

    def forward(self, *inputs):
        for op in self.outputs:
            op.reset("fwd")
            op.grad.reset("fwd")
        if len(inputs):
            assert len(inputs) == len(self.inputs)
            for op in self.inputs:
                op.reset("fwd")
            for inp, input in zip(self.inputs, inputs):
                inp.setvalue(input)
        else:
            for inp in self.inputs:
                inp.reset("fwd")
        for op in self.outputs:
            op.forward()
        return [op.output for op in self.outputs]

    def backward(self, *grads):
        if len(grads):
            assert len(grads) == len(self.outputs)
            for out, grad in zip(self.outputs, grads):
                out.grad.setvalue(grad)
        for param in self.parameters:
            param.grad.forward()

    def __repr__(self):
        return f"NetStatic({self.inputs},{self.parameters},{self.outputs})"

    def __str__(self):
        return self.__repr__()

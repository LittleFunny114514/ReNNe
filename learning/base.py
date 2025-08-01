from .. import base, cfg, moduleblock
import random


np = cfg.np

DESCENT = base.input(-1)
ASCENT = base.input(1)


class Learner:
    def __init__(self, *args, **kwargs):
        self.model: moduleblock.Module = None
        self.optimizer: base.Optimizer = None
        self.de_a: base.Data = ASCENT
        # self.lossfunc(x: base.Operation) -> base.Operation
        self.init(*args, **kwargs)
        self.model.setOptimizer(self.optimizer)

    def init(self, *args, **kwargs):
        """
        Most properties should be init in this function
        """
        raise NotImplementedError

    def descent(
        self,
        y: tuple[base.Operation],
        y_dst,
        loss_kwargs=dict(),
    ):
        assert not cfg.NO_GRAD
        loss = base.Add(*[self.loss_func(yi, yi_dst) for yi, yi_dst in zip(y, y_dst)])
        loss.grad = self.de_a
        loss.forward()
        self.model.backward()
        self.model.calcGradients()

        self.model.update()
        return loss.output

    def train(self, *args, **kwarge):
        raise NotImplementedError

    def loss_func(
        self,
        y: tuple[base.Operation],
        y_dst,
        **loss_kwargs,
    ) -> base.Operation:
        raise NotImplementedError


class Supervised(Learner):
    def init(
        self,
        model: moduleblock.Module,
        optimizer: base.Optimizer,
        model_batch_process=False,
    ):
        self.model: moduleblock.Module = model
        self.model_batch_process = model_batch_process
        self.optimizer: base.Optimizer = optimizer
        self.de_a: base.Data = DESCENT
        self.model.setOptimizer(self.optimizer)

    def loss_func(self, y: base.Operation, y_dst: base.Operation) -> base.Operation:
        raise NotImplementedError

    def train(
        self,
        dataset_x: list[np.ndarray],
        dataset_y: list[np.ndarray],
        total_batch: int,
        lr=0.001,
        batch_size=1,
    ):
        assert batch_size <= dataset_x[0].shape[0]
        assert batch_size > 0
        assert total_batch > 0
        assert not cfg.NO_GRAD
        for x, ydst in zip(dataset_x, dataset_y):
            assert x.shape[0] == ydst.shape[0] == dataset_x[0].shape[0]
        self.optimizer.lr = lr
        modelIshape, modelOshape = self.model.io_shape
        modelIcnt = len(self.model.io_shape[0])
        modelOcnt = len(self.model.io_shape[1])
        loss = 0.0
        last_timestamp = 0
        for timestamp in range(total_batch):
            self.model.clearOperationCorrelates()
            if self.model_batch_process:
                idx = random.sample(range(dataset_x[0].shape[0]), batch_size)
                x = [
                    base.input(
                        np.concatenate(
                            [dataset_xj[i].reshape((1, -1)) for i in idx], axis=0
                        )
                    )
                    for dataset_xj in dataset_x
                ]
                ydst = [
                    np.concatenate(
                        [dataset_yj[i].reshape((1, -1)) for i in idx], axis=0
                    )
                    for dataset_yj in dataset_y
                ]
                y = self.model.forward(*x)
                loss += self.descent(y, ydst)
                if timestamp % 100 == 0 or timestamp < 50:
                    print(
                        f"Batch {last_timestamp}~{timestamp} avg loss {loss/(timestamp-last_timestamp)}"
                    )
                    last_timestamp = timestamp
                    loss = 0.0
            else:
                idx = random.sample(range(dataset_x[0].shape[0]), batch_size)
                x = [None] * batch_size
                y = [None] * batch_size
                y_dst = [None] * batch_size
                for i in range(batch_size):
                    actual_index = idx[i]
                    x[i] = [
                        base.input(dataset_xj[actual_index]) for dataset_xj in dataset_x
                    ]
                    y[i] = self.model.forward(*(x[i]))
                    y_dst[i] = [dataset_yj[actual_index] for dataset_yj in dataset_y]
                y_flat = [None] * batch_size * modelOcnt
                y_dst_flat = [None] * batch_size * modelOcnt
                for i in range(batch_size):
                    for j in range(modelOcnt):
                        y_flat[i * modelOcnt + j] = y[i][j]
                        y_dst_flat[i * modelOcnt + j] = y_dst[i][j]
                loss += self.descent(y_flat, y_dst_flat)
                if timestamp % 100 == 0 or timestamp < 50:
                    print(
                        f"Batch {last_timestamp}~{timestamp} avg loss {loss/(timestamp-last_timestamp)}"
                    )
                    last_timestamp = timestamp
                    loss = 0.0

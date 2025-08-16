import ReNNe
import numpy as np
import tqdm
from ReNNe.datasets import MNIST


class TrainClassifier(ReNNe.learning.Supervised):
    def loss_func(self, y: ReNNe.Operation, y_dst: np.ndarray) -> ReNNe.Operation:
        return 0 + -ReNNe.input(y_dst) * ReNNe.Ln(y)


nn = ReNNe.moduleblock.Sequential(
    ReNNe.moduleblock.Lambda(ReNNe.Reshape, shape=(1, 28, 28)),
    ReNNe.conv.m.Conv2DDS(1, 16, 3),  # fm shape: (26, 26)
    ReNNe.conv.m.Pooling2D(2),  # fm shape: (13, 13)
    ReNNe.conv.m.Conv2DDS(16, 24, 3),  # fm shape: (11, 11)
    ReNNe.moduleblock.Lambda(ReNNe.Reshape, shape=(-1,)),
    ReNNe.moduleblock.Linear(24 * 11 * 11, 128),
    ReNNe.moduleblock.Linear(128, 10, activation=ReNNe.misc.Softmax),
)
trainer = TrainClassifier(nn, ReNNe.adamW(0.001, weight_decay=0.2))
train_x, train_label = MNIST.loadTrain()
train_x = MNIST.normalize(train_x)
train_y = MNIST.toOneHot(train_label)
test_x, test_label = MNIST.loadTest()
test_x, test_y = MNIST.normalize(test_x), MNIST.toOneHot(test_label)
print(nn.toString())


def train():
    nn.load("mnistconv.nn.npz")
    trainer.train([train_x], [train_y], 1001, batch_size=64)
    nn.save("mnistconv.nn")


def test():
    nn.load("mnistconv.nn.npz")
    y_train = np.zeros((train_x.shape[0], 10), dtype=ReNNe.cfg.dtype)
    y_test = np.zeros((test_x.shape[0], 10), dtype=ReNNe.cfg.dtype)
    for i in tqdm.tqdm(range(train_x.shape[0]), desc="Evaluating train set: "):
        y_train[i, :] = nn(train_x[i])[0]
    for i in tqdm.tqdm(range(test_x.shape[0]), desc="Evaluating test set"):
        y_test[i, :] = nn(test_x[i])[0]
    pred_train = np.argmax(y_train, axis=1)
    pred_test = np.argmax(y_test, axis=1)
    train_acc = np.sum((pred_train == train_label).astype(int)) / train_label.shape[0]
    test_acc = np.sum((pred_test == test_label).astype(int)) / test_label.shape[0]
    print("train acc:", train_acc)
    print("test acc:", test_acc)

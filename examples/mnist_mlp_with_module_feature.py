import ReNNe
import numpy as np
from ReNNe.datasets import MNIST


class TrainClassifier(ReNNe.learning.Supervised):
    def loss_func(self, y: ReNNe.Operation, y_dst: np.ndarray) -> ReNNe.Operation:
        return 0 + -ReNNe.input(y_dst) * ReNNe.Ln(y)


nn = ReNNe.moduleblock.Sequential(
    ReNNe.moduleblock.Linear(784, 128),
    ReNNe.moduleblock.Linear(128, 10, activation=ReNNe.Softmax),
)
trainer = TrainClassifier(nn, ReNNe.adamW(0.001))
train_x, train_label = MNIST.loadTrain()
train_x = MNIST.normalize(train_x).reshape(-1, 784)
train_y = MNIST.toOneHot(train_label)
test_x, test_label = MNIST.loadTest()
test_x, test_y = MNIST.normalize(test_x).reshape(-1, 784), MNIST.toOneHot(test_label)
print(nn.toString())


def train():
    trainer.train([train_x], [train_y], 2000, batch_size=64)
    nn.save("mnistmlpmod.nn")


def test():
    nn.load("mnistmlpmod.nn.npz")
    (y_train,) = nn(train_x)
    (y_test,) = nn(test_x)
    pred_train = np.argmax(y_train, axis=1)
    pred_test = np.argmax(y_test, axis=1)
    train_acc = np.sum((pred_train == train_label).astype(int)) / train_label.shape[0]
    test_acc = np.sum((pred_test == test_label).astype(int)) / test_label.shape[0]
    print("train acc:", train_acc)
    print("test acc:", test_acc)

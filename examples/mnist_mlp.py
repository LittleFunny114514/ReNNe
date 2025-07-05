import numpy as np
import random
import time

import ReNNe
from ..datasets import MNIST


class NN:
    def __init__(
        self,
        hiddenneurons,
        optimizer=ReNNe.WeightDecay(0.1, ReNNe.Adam(0.0005, 0.8, 0.99)),
    ):
        self.hiddenneurons = hiddenneurons
        self.w1 = ReNNe.parameter((784, hiddenneurons))
        self.b1 = ReNNe.parameter((hiddenneurons,))
        self.w2 = ReNNe.parameter((hiddenneurons, 10))
        self.b2 = ReNNe.parameter((10,))

        self.input = ReNNe.input(np.zeros((1, 784)))
        self.l1 = ReNNe.Tanh(ReNNe.Matmul(self.input, self.w1) + self.b1)
        self.l2 = ReNNe.Softmax(ReNNe.Matmul(self.l1, self.w2) + self.b2)

        self.net = ReNNe.NetStatic()
        self.net.register("parameter", self.w1, self.b1, self.w2, self.b2)
        self.net.register("input", self.input)
        self.net.register("output", self.l2)

        self.net.setOptimizer(optimizer)
        self.net.initParameters()
        self.net.build()
        self.loss = ReNNe.CrossEntropy(self.net)

    def forward(self, x):
        self.l2.reset("fwd")
        self.input.setvalue(x)
        self.net.forward()
        return self.l2.output

    def train(self, x, ydst, epochs, batchsize=64):
        for epoch in range(epochs):
            idxbegin = random.randint(0, len(x) - batchsize)
            x_batch = x[idxbegin : idxbegin + batchsize, :]
            ydst_batch = ydst[idxbegin : idxbegin + batchsize, :]
            loss, accuracy = self.loss.fit([x_batch], [ydst_batch])
            self.net.update()
            if epoch % 100 == 0:
                # time.sleep(1)
                # print(self.b1.value,'\n',self.b2.value)
                print(f"Epoch {epoch} loss: {loss} accuracy: {accuracy}")

    def evaluate(self, x, ydst):
        y = self.forward(x)
        loss, accuracy = self.loss.evaluate([y], [ydst])
        print(y[:10], ydst[:10])
        print(f"Test loss: {loss} accuracy: {accuracy}")
        return loss, accuracy


hiddenneurons = 128


def train(model_save_path):
    images, labels = MNIST.loadTrain()
    images = MNIST.normalize(images)
    labels = MNIST.toOneHot(labels)
    images = images.reshape(-1, 784)
    nn = NN(hiddenneurons)
    nn.train(images, labels, epochs=5000)
    nn.evaluate(images, labels)
    nn.net.save(model_save_path)


def test(model_save_path):
    images, labels = MNIST.readTest()
    images = MNIST.normalize(images)
    labels = MNIST.toOneHot(labels)
    images = images.reshape(-1, 784)
    nn = NN(hiddenneurons)
    nn.net.load(model_save_path)
    nn.evaluate(images, labels)


def ui(model_save_path):
    import pygame

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 20)
    screen = pygame.display.set_mode((280, 280))
    pygame.display.set_caption("MNIST UI")
    nn = NN(hiddenneurons)
    nn.net.load(model_save_path)
    img = np.zeros((28, 28))
    clock = pygame.time.Clock()
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        mousepressed = pygame.mouse.get_pressed()
        if mousepressed[0] or mousepressed[2]:
            # right click to clear
            if mousepressed[2]:
                img = np.zeros((28, 28))
            else:
                # left click to draw
                x, y = pygame.mouse.get_pos()
                if 0 <= x < 280 and 0 <= y < 280:
                    x //= 10
                    y //= 10
                    img[y, x] = 1
            output = nn.forward(img.reshape(-1, 784))
            predict = np.argmax(output)
            # display image
            for i in range(28):
                for j in range(28):
                    if img[j, i] == 1:
                        pygame.draw.rect(
                            screen, (255, 255, 255), (i * 10, j * 10, 10, 10)
                        )
                    else:
                        pygame.draw.rect(screen, (0, 0, 0), (i * 10, j * 10, 10, 10))
            text = font.render(
                f"Predict: {predict},{int(output[0,predict]*100)}%",
                True,
                (255, 255, 255),
            )
            screen.blit(text, (0, 0))
            pygame.display.update()

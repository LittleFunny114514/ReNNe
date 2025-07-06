import matplotlib.pyplot as plt
from .. import cfg, base

np = cfg.np


def read(pathimages, pathlabels):
    with open(pathimages, "rb") as f:
        # 读取文件头信息
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_images = int.from_bytes(f.read(4), byteorder="big")
        rows = int.from_bytes(f.read(4), byteorder="big")
        cols = int.from_bytes(f.read(4), byteorder="big")

        # 读取图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    with open(pathlabels, "rb") as f:
        magic_number = int.from_bytes(f.read(4), byteorder="big")
        num_labels = int.from_bytes(f.read(4), byteorder="big")

        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels


def toOneHot(labels: np.ndarray):
    onehot = np.zeros((labels.shape[0], 10), dtype=cfg.dtype)
    for i in range(labels.shape[0]):
        onehot[i][labels[i]] = 1
    return onehot


def normalize(images: np.ndarray):
    return images.astype(cfg.dtype) / 255


def loadTrain():
    return read(
        "ReNNe/datasets/train-images.idx3-ubyte",
        "ReNNe/datasets/train-labels.idx1-ubyte",
    )


def loadTest():
    return read(
        "ReNNe/datasets/t10k-images.idx3-ubyte", "ReNNe/datasets/t10k-labels.idx1-ubyte"
    )


def show(images, labels):
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    for i in range(2):
        for j in range(5):
            axs[i, j].imshow(images[i * 5 + j], cmap="gray")
            axs[i, j].set_title("label:" + str(labels[i * 5 + j]))
            axs[i, j].axis("off")
    plt.show()

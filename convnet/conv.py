from .. import base, cfg, np, misc


def conv2d(fm: np.ndarray, krnls: np.ndarray, padh=0, padw=None):
    padw = padh if padw is None else padw
    assert fm.ndim == krnls.ndim == 3
    channels = krnls.shape[0]
    kernelw = krnls.shape[2]
    kernelh = krnls.shape[1]
    fmw = fm.shape[2]
    fmh = fm.shape[1]
    assert channels == fm.shape[0]
    padded = (
        np.pad(fm, ((0, 0), (padh, padh), (padw, padw)), "constant", constant_values=0)
        if max(padh, padw) > 0
        else fm
    )

    ret = np.zeros(
        (channels, fmh - kernelh + 1 + 2 * padh, fmw - kernelw + 1 + 2 * padw),
        dtype=cfg.dtype,
    )

    for i in range(channels):
        ret[i, :, :] = cfg.sp.signal.correlate2d(
            padded[i, :, :], krnls[i, :, :], "valid"
        )

    if padh < 0:
        ret = ret[:, -padh + 1 : padh, :]
    if padw < 0:
        ret = ret[:, :, -padw + 1 : padw]
    return ret


def conv2dds(
    weight: base.Operation, kernels: base.Operation, input: base.Operation, padding=0
):
    assert (
        input.output.ndim == 3 and kernels.output.ndim == 3 and weight.output.ndim == 2
    )
    assert input.output.shape[0] == weight.output.shape[0]
    assert weight.output.shape[1] == kernels.output.shape[0]
    return conv2ddeepwise(weight, Conv2dSpare(kernels, input, padding))


def conv2ddeepwise(weight: base.Operation, input: base.Operation):
    assert input.output.ndim == 3 and weight.output.ndim == 2
    assert input.output.shape[0] == weight.output.shape[1]
    return misc.Einsum("oc,chw->ohw", weight, input)


class Conv2dSpare(base.Operation):
    def __init__(self, kernels: base.Operation, input: base.Operation, padding=0):
        self.kernels = kernels
        self.input = input
        self.padding = padding
        super().__init__(kernels, input)

    def forwardUnwrap(self):
        self.output = conv2d(self.input.output, self.kernels.output, self.padding)

    def backwardUnwrap(self):
        channels = self.kernels.output.shape[0]
        kernelw = self.kernels.output.shape[2]
        kernelh = self.kernels.output.shape[1]
        fmw = self.input.output.shape[2]
        fmh = self.input.output.shape[1]

        kernelinv = np.flip(self.kernels.output, axis=(1, 2))

        self.kernels.grad += conv2d(self.input.output, self.grad)
        self.input.grad += conv2d(
            self.grad, kernelinv, kernelh - 1 - self.padding, kernelw - 1 - self.padding
        )

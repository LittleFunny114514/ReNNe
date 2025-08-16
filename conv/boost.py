from typing import Literal
from ..cfg import np, nb, sci
from .. import cfg
import math


def _():
    print("ReNNe: Numba compiler: Compiling conv2db...")

    @nb.jit(
        [
            nb.float32[:, :, :](
                nb.float32[:, :, :], nb.float32[:, :, :], nb.uintp, nb.uintp
            ),
            nb.float64[:, :, :](
                nb.float64[:, :, :], nb.float64[:, :, :], nb.uintp, nb.uintp
            ),
        ],
        nopython=True,
        cache=True,
        parallel=False,
    )
    def conv2db_njit(
        fm: np.ndarray,
        krnl: np.ndarray,
        padx: int,
        pady: int,
    ) -> np.ndarray:
        assert fm.shape[0] == krnl.shape[0]
        channels, krnl_rows, krnl_cols = krnl.shape
        output_rows = fm.shape[1] - krnl.shape[1] + 2 * padx + 1
        output_cols = fm.shape[2] - krnl.shape[2] + 2 * pady + 1
        maxx0 = max(padx, 0)
        maxy0 = max(pady, 0)
        minx0 = min(padx, 0)
        miny0 = min(pady, 0)

        ret: np.ndarray = np.zeros((channels, output_rows, output_cols), dtype=fm.dtype)
        padfm: np.ndarray = np.zeros(
            (channels, fm.shape[1] + 2 * padx, fm.shape[2] + 2 * pady), fm.dtype
        )
        padfm[:, maxx0 : -maxx0 + padfm.shape[1], maxy0 : -maxy0 + padfm.shape[2]] = fm[
            :, -minx0 : minx0 + fm.shape[1], -miny0 : miny0 + fm.shape[2]
        ]

        for c, i, j in np.ndindex(ret.shape):
            sum_val = 0.0
            for ki in range(krnl_rows):
                for kj in range(krnl_cols):
                    sum_val += padfm[c, i + ki, j + kj] * krnl[c, ki, kj]
            ret[c, i, j] = sum_val

        return ret

    def wrapper_njit(
        fm: np.ndarray,
        krnl: np.ndarray,
        padx: int = 0,
        pady: int = None,
    ) -> np.ndarray:
        if pady is None:
            pady = padx
        return conv2db_njit(fm, krnl, padx, pady)

    return wrapper_njit


conv2db = _()


def _():
    print("ReNNe: Numba compiler: Compiling maxpooling2db...")

    @nb.jit(
        [
            nb.float32[:, :, :](nb.float32[:, :, :], nb.uintp, nb.uintp),
            nb.float64[:, :, :](nb.float64[:, :, :], nb.uintp, nb.uintp),
        ],
        nopython=True,
        cache=True,
        parallel=False,
    )
    def maxpooling2db_njit(fm: np.ndarray, size: int, stride: int = 0) -> np.ndarray:

        channels, fm_rows, fm_cols = fm.shape
        ret_rows = math.ceil(fm.shape[1] / stride)
        ret_cols = math.ceil(fm.shape[2] / stride)
        padfm_rows = stride * (ret_rows - 1) + size
        padfm_cols = stride * (ret_cols - 1) + size

        padfm = -math.inf * np.ones((channels, padfm_rows, padfm_cols), dtype=fm.dtype)
        padfm[:, 0:fm_rows, 0:fm_cols] = fm
        ret = np.zeros((channels, ret_rows, ret_cols), dtype=fm.dtype)
        for c, I, J in np.ndindex(ret.shape):
            i, j = I * stride, J * stride
            maxv = -math.inf
            for iblock in range(size):
                for jblock in range(size):
                    maxv = max(maxv, padfm[c, i + iblock, j + jblock])
            ret[c, I, J] = maxv
        return ret

    @nb.jit(
        [
            nb.float32[:, :, :](nb.float32[:, :, :], nb.uintp, nb.uintp),
            nb.float64[:, :, :](nb.float64[:, :, :], nb.uintp, nb.uintp),
        ],
        nopython=True,
        cache=True,
        parallel=False,
    )
    def avgpooling2db_njit(fm: np.ndarray, size: int, stride: int = 0) -> np.ndarray:

        channels, fm_rows, fm_cols = fm.shape
        ret_rows = math.ceil(fm.shape[1] / stride)
        ret_cols = math.ceil(fm.shape[2] / stride)

        ret = np.zeros((channels, ret_rows, ret_cols), dtype=fm.dtype)
        for c, I, J in np.ndindex(ret.shape):
            i, j = I * stride, J * stride
            sum = 0.0
            block_w = min(size, fm_rows - i)
            block_h = min(size, fm_cols - j)
            block_size = block_w * block_h
            for iblock in range(block_w):
                for jblock in range(block_h):
                    sum += fm[c, i + iblock, j + jblock]
            ret[c, I, J] = sum / block_size
        return ret

    def wrapper(
        fm: np.ndarray, size: int, stride: int = 0, mode: Literal["max", "avg"] = "max"
    ) -> np.ndarray:
        assert size > 0
        if stride == 0:
            stride = size
        if mode == "max":
            return maxpooling2db_njit(fm, size, stride)
        elif mode == "avg":
            return avgpooling2db_njit(fm, size, stride)
        else:
            raise NotImplementedError

    return wrapper


pooling2db = _()


def _():
    print("ReNNe: Numba compiler: Compiling bwdmaxpooling2db...")

    @nb.jit(
        [
            nb.float32[:, :, :](
                nb.float32[:, :, :], nb.float32[:, :, :], nb.uintp, nb.uintp
            ),
            nb.float64[:, :, :](
                nb.float64[:, :, :], nb.float64[:, :, :], nb.uintp, nb.uintp
            ),
        ],
        nopython=True,
        cache=True,
        parallel=False,
    )
    def bwdmaxpooling2db_njit(
        fm: np.ndarray, outgrad: np.ndarray, size: int, stride: int
    ):
        channels, output_rows, output_cols = outgrad.shape
        channels, fm_rows, fm_cols = fm.shape
        padfm_rows = stride * (output_rows - 1) + size
        padfm_cols = stride * (output_cols - 1) + size

        padfm = -math.inf * np.ones((channels, padfm_rows, padfm_cols), dtype=fm.dtype)
        padfm[:, 0:fm_rows, 0:fm_cols] = fm
        ret = np.zeros_like(fm)
        for c, I, J in nb.pndindex(outgrad.shape):
            i, j = I * stride, J * stride
            maxv = -math.inf
            for iblock in range(size):
                for jblock in range(size):
                    maxv = max(maxv, padfm[c, i + iblock, j + jblock])
            for iblock in range(size):
                for jblock in range(size):
                    if maxv == padfm[c, i + iblock, j + jblock]:
                        ret[c, i + iblock, j + jblock] = outgrad[c, I, J]
        return ret

    def wrapper(fm: np.ndarray, outgrad: np.ndarray, size: int, stride: int = None):
        if stride is None:
            stride = size
        assert np.isfinite(fm).all()
        return bwdmaxpooling2db_njit(fm, outgrad, size, stride)

    return wrapper


bwdmaxpooling2db = _()

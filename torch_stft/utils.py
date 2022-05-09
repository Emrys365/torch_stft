# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from distutils.version import LooseVersion
from itertools import chain
import math

import torch
import torch.nn.functional as F


is_torch_1_8_plus = LooseVersion(torch.__version__) >= LooseVersion("1.8.0")


def get_window(window, win_length, **kwargs):
    if window is None:
        return window

    elif callable(window):
        return window(win_length)

    elif isinstance(window, str):
        if hasattr(torch, f"{window}_window"):
            return getattr(torch, f"{window}_window")(win_length, **kwargs)

        raise ValueError(f"Unknown window type: {window}")

    elif isinstance(window, (torch.Tensor, list)):
        if len(window) == win_length:
            return torch.as_tensor(window, **kwargs)

        raise ValueError(f"Window size mismatch: {len(window)} != {win_length}")
    else:
        raise ValueError(f"Invalid window specification: {window}")


def pad_to_size(data, size, dim=-1, pad_type="center", **kwargs):
    kwargs.setdefault("mode", "constant")
    assert pad_type in ("left", "center", "right"), pad_type

    n = data.size(dim)
    if n == size:
        return data

    pad = size - n

    lengths = [(0, 0)] * data.dim()
    if pad_type == "left":
        lengths[dim] = (pad, 0)
    elif pad_type == "center":
        lpad = int(pad // 2)
        lengths[dim] = (lpad, int(size - n - lpad))
    elif pad_type == "right":
        lengths[dim] = (0, pad)
    lengths = tuple(chain(*reversed(lengths)))

    if pad < 0:
        raise ValueError(
            ("Target size ({:d}) must be " "at least input size ({:d})").format(size, n)
        )

    return F.pad(data, lengths, **kwargs)


def frame(x, frame_length, hop_length, dim=-1):
    # import librosa
    # import numpy as np
    # np.testing.assert_allclose(
    #     frame(x, frame_length, hop_length, dim=-1).numpy(),
    #     librosa.util.frame(x.numpy(), frame_length, hop_length, axis=-1),
    # )
    if x.size(dim) < frame_length:
        raise ValueError(
            "Input is too short (n={:d})"
            " for frame_length={:d}".format(x.size(dim), frame_length)
        )

    if hop_length < 1:
        raise ValueError("Invalid hop_length: {:d}".format(hop_length))

    n_frames = 1 + (x.size(dim) - frame_length) // hop_length
    strides = torch.as_tensor(x.stride())

    if dim == -1 or dim == x.dim() - 1:
        shape = list(x.shape[:-1]) + [frame_length, n_frames]
        strides = list(strides) + [hop_length]

    elif dim == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length] + list(strides)

    else:
        raise ValueError("Frame dim={} must be either 0 or -1".format(dim))

    return x.as_strided(shape, strides, storage_offset=0)


def get_dft_bases(n_fft, round_pow_of_two=True):
    # Ref: https://en.wikipedia.org/wiki/DFT_matrix#Definition
    # FFT points
    N = 2 ** math.ceil(math.log2(n_fft)) if round_pow_of_two else n_fft
    # DFT{ δ[n - n0] } = exp(-j 2π k n0 / N), where N is n_fft
    if is_torch_1_8_plus:
        delayed_delta = torch.eye(N)
        # (n_fft, N, 2)
        dft_bases = torch.view_as_real(torch.fft.fft(delayed_delta))
    else:
        delayed_delta = torch.stack([torch.eye(N), torch.zeros(N, N)], dim=-1)
        # (n_fft, N, 2)
        dft_bases = torch.fft(delayed_delta, 1)[:n_fft]
    return dft_bases


def tiny(x):
    # Make sure we have an array view
    x = torch.as_tensor(x)

    # Only floating types generate a tiny
    if torch.is_floating_point(x) or (
        is_torch_1_8_plus and torch.is_complex(x) and torch.is_floating_type(x.real)
    ):
        dtype = x.dtype
    else:
        dtype = torch.float32

    return torch.finfo(dtype).tiny


def fix_length(data, size, dim=-1):
    n = data.size(dim)

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[dim] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        return pad_to_size(data, size, dim=dim, pad_type="right")

    return data

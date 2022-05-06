from functools import partial
import time
import warnings

import numpy as np
import pytest
import soundfile as sf
import torch

from torch_stft import ShortTimeFourierTransform
from torch_stft.utils import get_window


def test_speed(func, *args, num_runs=100, **kwargs):
    ret = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        ret.append((end - start) * 1000)

    mean, std = np.mean(ret), np.std(ret)
    info = "{:.1f} ms ± {:.1f} ms per loop (mean ± std. dev. of {} runs, 1 loop each"
    print(info.format(mean, std, num_runs))
    return mean, std


@pytest.mark.parametrize(
    "n_fft, hop_length, win_length", [(512, 256, 400), (1024, 256, 1024)]
)
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect"])
@pytest.mark.parametrize("normalized", [True])
@pytest.mark.parametrize("onesided", [True, False])
@pytest.mark.parametrize("num_runs", [100])
def test_stft_speed(
    n_fft,
    hop_length,
    win_length,
    window,
    center,
    pad_mode,
    normalized,
    onesided,
    num_runs,
):
    wav, fs = sf.read("tests/test.wav", always_2d=True)
    x = torch.from_numpy(wav[:, 0])
    elapsed_time = {}
    for impl in ("fft", "matmul", "conv"):
        stft = ShortTimeFourierTransform(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            impl=impl,
        )
        elapsed_time[impl] = test_speed(stft, x, num_runs=num_runs)[0]

    win = get_window(window, win_length, periodic=True)
    elapsed_time["native"] = test_speed(
        partial(
            torch.stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=win,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
        ),
        x,
        num_runs=num_runs,
    )[0]

    if elapsed_time["native"] > elapsed_time["fft"]:
        info = "'fft' based implementation ({:.1f} ms) is faster than "
        "`torch.stft` ({:.1f} ms)"
        warnings.warn(info.format(elapsed_time["fft"], elapsed_time["native"]))
    else:
        assert (
            elapsed_time["native"]
            < elapsed_time["fft"]
            < elapsed_time["matmul"]
            < elapsed_time["conv"]
        )


@pytest.mark.parametrize(
    "n_fft, hop_length, win_length", [(512, 256, 400), (1024, 256, 1024)]
)
@pytest.mark.parametrize("window", ["hann"])
@pytest.mark.parametrize("center", [True])
@pytest.mark.parametrize("pad_mode", ["reflect"])
@pytest.mark.parametrize("normalized", [True])
@pytest.mark.parametrize("onesided", [True, False])
@pytest.mark.parametrize("num_runs", [100])
def test_istft_speed(
    n_fft,
    hop_length,
    win_length,
    window,
    center,
    pad_mode,
    normalized,
    onesided,
    num_runs,
):
    wav, fs = sf.read("tests/test.wav", always_2d=True)
    x = torch.from_numpy(wav[:, 0])
    elapsed_time = {}
    for impl in ("fft", "matmul", "conv"):
        stft = ShortTimeFourierTransform(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            impl=impl,
        )
        spec = stft(x)
        istft = partial(stft.inverse, length=len(x))
        elapsed_time[impl] = test_speed(istft, spec, num_runs=num_runs)[0]

    win = get_window(window, win_length, periodic=True)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=win,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=onesided,
    )
    elapsed_time["native"] = test_speed(
        partial(
            torch.istft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=win,
            center=center,
            normalized=normalized,
            onesided=onesided,
        ),
        spec,
        num_runs=num_runs,
    )[0]

    if elapsed_time["native"] > elapsed_time["fft"]:
        info = "'fft' based implementation ({:.1f} ms) is faster than "
        "`torch.istft` ({:.1f} ms)"
        warnings.warn(info.format(elapsed_time["fft"], elapsed_time["native"]))
    else:
        assert (
            elapsed_time["native"]
            < elapsed_time["fft"]
            < elapsed_time["matmul"]
            < elapsed_time["conv"]
        )

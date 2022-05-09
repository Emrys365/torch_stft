import numpy as np
import pytest
import soundfile as sf
import torch

from torch_stft import ShortTimeFourierTransform
from torch_stft.utils import get_window


is_cuda_available = torch.cuda.is_available()


@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("hop_length", [None, 256])
@pytest.mark.parametrize("win_length", [None, 400])
@pytest.mark.parametrize("window", ["hann", "hamming"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant"])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
@pytest.mark.parametrize("impl", ["fft", "matmul", "conv"])
def test_stft_istft_consistency(
    n_fft,
    hop_length,
    win_length,
    window,
    center,
    pad_mode,
    normalized,
    onesided,
    impl,
):
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
    wav, fs = sf.read("tests/test.wav", always_2d=True)
    x = torch.from_numpy(wav[:, 0])
    if is_cuda_available:
        stft.cuda()
        x = x.cuda()

    if center:
        same_length = True
    else:
        if win_length is None:
            win_length = n_fft
        win = get_window(window, win_length, periodic=True)
        if is_cuda_available:
            win = win.cuda()
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
            return_complex=False,
        )
        try:
            x2 = torch.istft(
                spec,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=win,
                center=center,
                normalized=normalized,
                onesided=onesided,
                length=len(x),
            )
        except RuntimeError:
            # RuntimeError: istft(...) window overlap add min: 0
            x2 = torch.zeros(0, device=x.device)
        same_length = len(x2) == len(x)

    seg = slice(None) if center else slice(stft.hop_length, -stft.hop_length)
    if same_length and center:
        np.testing.assert_allclose(
            stft.inverse(stft(x), len(x))[seg].cpu().numpy(),
            x[seg].cpu().numpy(),
            atol=5e-04,
        )
        x2 = torch.stack([x, x], dim=0)
        np.testing.assert_allclose(
            stft.inverse(stft(x2), len(x))[:, seg].cpu().numpy(),
            x2[:, seg].cpu().numpy(),
            atol=5e-04,
        )
    else:
        x3 = stft.inverse(stft(x), len(x))
        np.testing.assert_allclose(
            x3[seg].cpu().numpy(), x[: len(x3)][seg].cpu().numpy(), atol=8e-04
        )
        x2 = torch.stack([x, x], dim=0)
        x3 = stft.inverse(stft(x2), len(x))
        np.testing.assert_allclose(
            x3[:, seg].cpu().numpy(),
            x2[:, : x3.size(1)][:, seg].cpu().numpy(),
            atol=8e-04,
        )


@pytest.mark.parametrize("n_fft", [512])
@pytest.mark.parametrize("hop_length", [None, 256])
@pytest.mark.parametrize("win_length", [None, 400])
@pytest.mark.parametrize("window", ["hann", "hamming"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant"])
@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("onesided", [True, False])
def test_stft_consistency(
    n_fft,
    hop_length,
    win_length,
    window,
    center,
    pad_mode,
    normalized,
    onesided,
):
    wav, fs = sf.read("tests/test.wav", always_2d=True)
    x = torch.from_numpy(wav[:, 0])
    if is_cuda_available:
        x = x.cuda()

    specs = []
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
        if is_cuda_available:
            stft.cuda()
        specs.append(stft(x))

    for s in specs[1:]:
        np.testing.assert_allclose(specs[0].cpu().numpy(), s.cpu().numpy(), atol=1e-06)

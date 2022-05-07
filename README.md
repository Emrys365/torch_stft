# PyTorch-based implementations of STFT and iSTFT

This repository provides three variants of STFT/iSTFT implementations based on PyTorch:
* `fft`: FFT based implementation
* `matmul`: matrix multiplication based implementation
* `conv`: Conv1D based implementation

> `fft` based implementation is the fastest among the three, but is still a bit slower than the native implementation in PyTorch.

All implementations support inputs with and without a batch dimension.

## Install

```bash
# install via git
python -m pip install git+https://github.com/Emrys365/torch_stft

# install from source
git clone git@github.com:Emrys365/torch_stft.git
cd torch_stft
python -m pip install -e .
```

## Usage

```python
import torch
from torch_stft import ShortTimeFourierTransform

device = "cpu"
stft = ShortTimeFourierTransform(
    n_fft=512,
    win_length=400,
    hop_length=256,
    center=True,
    pad_mode="reflect",
    normalized=True,
    onesided=True,
    impl="fft",
).to(device)

signal = torch.rand(2, 16000, device=device)
spec = stft(signal, return_complex=False)   # (2, 257, 63, 2)
spec = stft(signal, return_complex=True)    # (2, 257, 63)
signal_recon = stft.inverse(spec, length=signal.shape[1])
assert torch.allclose(signal, signal_recon, atol=1e-6)
```

## Test implementations

```bash
python -m pytest tests/
```

## References

[1] https://en.wikipedia.org/wiki/DFT_matrix

[2] https://www.funcwj.cn/2020/04/10/conv1d-ops/

[3] https://librosa.org/doc/latest/_modules/librosa/core/spectrum.html#stft

[4] https://github.com/pytorch/audio/blob/2ebbbf511fb1e6c47b59fd32ad7e66023fa0dff1/torchaudio/functional.py#L44

# PyTorch-based implementations of STFT and iSTFT

This repository provides three variants of STFT/iSTFT implementations based on PyTorch:
* `fft`: FFT based implementation
* `matmul`: matrix multiplication based implementation
* `conv`: Conv1D based implementation

All implementations support inputs with and without a batch dimension.

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

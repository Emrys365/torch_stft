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
import math
from typing import Callable
from typing import Optional
from typing import Union
import warnings

import torch
import torch.nn.functional as F

from .utils import frame
from .utils import get_dft_bases
from .utils import get_window
from .utils import pad_center
from .utils import tiny


is_torch_1_8_plus = LooseVersion(torch.__version__) >= LooseVersion("1.8.0")


class ShortTimeFourierTransform(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: Optional[int] = 128,
        win_length: Optional[int] = None,
        window: Union[Callable, str, list, torch.Tensor, None] = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
        impl: str = "fft",
    ):
        """Implementation of Short-Time Fourier Transform (STFT) and its inverse transform.

        Args:
            n_fft (int): size of Fourier transform
            hop_length (int, optional): the distance between neighboring sliding window
                frames. Default: ``None`` (treated as equal to ``floor(n_fft / 4)``)
            win_length (int, optional): the size of window frame and STFT filter.
                Default: ``None``  (treated as equal to :attr:`n_fft`)
            window (Tensor, optional): the optional window function.
                Default: ``None`` (treated as window of all :math:`1` s)
            center (bool, optional): whether to pad :attr:`input` on both sides so
                that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
                Default: ``True``
            pad_mode (string, optional): controls the padding method used when
                :attr:`center` is ``True``. Default: ``"reflect"``
            normalized (bool, optional): controls whether to return the normalized STFT results
                Default: ``False``
            onesided (bool, optional): controls whether to return half of results to
                avoid redundancy for real inputs.
                Default: ``True`` for real :attr:`input` and :attr:`window`, ``False`` otherwise.
            impl (str): specifies which implementation ("fft", "matmul", "conv") to use.
                Default: "fft".
        """  # noqa E501
        super().__init__()

        assert impl in ("fft", "matmul", "conv"), impl
        self.impl = impl

        self.n_fft = n_fft
        if win_length is None:
            win_length = n_fft
        self.win_length = win_length
        if hop_length is None:
            hop_length = int(win_length // 4)
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        if isinstance(window, str) and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

        # DFT basis (n_fft, N, 2)
        # This is a unitary matrix after divided by `sqrt(n_fft)`,
        # which means its conjugate transpose is also its inverse.
        dft_bases = get_dft_bases(n_fft, round_pow_of_two=False)
        half_F = n_fft // 2
        assert torch.allclose(
            dft_bases[:, 1 : 1 + half_F, 0],
            torch.flip(dft_bases[:, half_F:, 0], (1,)),
            atol=1e-06,
        ) and torch.allclose(
            dft_bases[:, 1 : 1 + half_F, 1],
            -torch.flip(dft_bases[:, half_F:, 1], (1,)),
            atol=1e-06,
        )
        self.cutoff = half_F + 1 if onesided else n_fft
        self.register_buffer("dft_bases", dft_bases)

        # Pad the window out to n_fft size
        fft_window = get_window(window, win_length, periodic=True)
        if fft_window is not None:
            fft_window = pad_center(fft_window, n_fft)
            self.normalized_scale = float(fft_window.pow(2).sum().sqrt())
        else:
            self.normalized_scale = math.sqrt(n_fft)
        self.register_buffer("dft_window", fft_window)

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"pad_mode='{self.pad_mode}', "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}, "
            f"impl='{self.impl}'"
        )

    def forward(self, input: torch.Tensor, return_complex: Optional[bool] = None):
        """Compute Short-time Fourier transform (STFT).

        The STFT computes the Fourier transform of short overlapping windows of the
        input. This giving frequency components of the signal as they change over
        time. The interface of this function is modeled after the librosa_ stft function:

        _librosa: https://librosa.org/doc/latest/generated/librosa.stft.html

        Args:
            input (Tensor): the input tensor (Batch, nSamples) or (nSamples,)
            return_complex (bool, optional): whether to return a complex tensor, or
                a real tensor with an extra last dimension for the real and
                imaginary components.

        Returns:
            stft_matrix (Tensor): STFT spectrum (Batch, Freq, Frame) or (Freq, Frame)
        """  # noqa E501
        # Check audio is valid
        assert input.dim() in (1, 2), input.dim()
        dtype = input.dtype

        has_batch_dim = input.dim() == 2
        fft_window = self.dft_window
        if fft_window is not None:
            # Reshape so that the window can be broadcast
            if has_batch_dim:
                fft_window = fft_window.to(dtype=dtype).reshape(1, -1, 1)
            else:
                fft_window = fft_window.to(dtype=dtype).reshape(-1, 1)

        # Pad the time series so that frames are centered
        if self.center:
            if self.n_fft > input.size(-1):
                info = "n_fft={} is too small for input signal of length={}"
                warnings.warn(info.format(self.n_fft, input.size(-1)))

            signal_dim = input.dim()
            extended_shape = [1] * (3 - signal_dim) + list(input.size())
            pad = int(self.n_fft // 2)
            input = F.pad(input.view(extended_shape), [pad, pad], self.pad_mode)
            input = input.view(input.shape[-signal_dim:])

        elif self.n_fft > input.size(-1):
            info = "n_fft={} is too large for input signal of length={}"
            raise ValueError(info.format(self.n_fft, input.shape[-1]))

        # Compute the STFT matrix ([B,] F, T, 2)
        if self.impl == "fft":
            stft_matrix = self._fft_stft(input, fft_window).transpose(-2, -3)
        elif self.impl == "matmul":
            stft_matrix = self._matmul_stft(input, fft_window).transpose(-2, -3)
        elif self.impl == "conv":
            stft_matrix = self._conv_stft(input, fft_window).transpose(-2, -3)

        if self.normalized:
            stft_matrix /= self.normalized_scale

        if return_complex:
            assert is_torch_1_8_plus
            # ([B,] F, T)
            return torch.view_as_complex(stft_matrix).contiguous()
        else:
            # ([B,] F, T, 2)
            return stft_matrix.contiguous()

    def inverse(self, stft_matrix, length=None):
        """Inverse short time Fourier Transform. This is expected to be the inverse of `forward`.

        Ported from torchaudio.functional.istft in v0.5.0

        It has the same parameters (+ additional optional parameter of :attr:`length`) and it should return the
        least squares estimation of the original signal. The algorithm will check using the NOLA condition (
        nonzero overlap).

        Important consideration in the parameters :attr:`window` and :attr:`center` so that the envelop
        created by the summation of all the windows is never zero at certain point in time. Specifically,
        :math:`\sum_{t=-\infty}^{\infty} |w|^2[n-t\times hop\_length] \cancel{=} 0`.

        Since `forward` discards elements at the end of the signal if they do not fit in a frame,
        ``inverse`` may return a shorter signal than the original signal (can occur if :attr:`center` is False
        since the signal isn't padded).

        If :attr:`center` is ``True``, then there will be padding e.g. ``'constant'``, ``'reflect'``, etc.
        Left padding can be trimmed off exactly because they can be calculated but right padding cannot be
        calculated without additional information.

        Example: Suppose the last window is:
        ``[17, 18, 0, 0, 0]`` vs ``[18, 0, 0, 0, 0]``

        The :attr:`n_fft`, :attr:`hop_length`, :attr:`win_length` are all the same which prevents the calculation
        of right padding. These additional values could be zeros or a reflection of the signal so providing
        :attr:`length` could be useful. If :attr:`length` is ``None`` then padding will be aggressively removed
        (some loss of signal).

        [1] D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984.

        Args:
            stft_matrix (Tensor): Output of stft where each row of a channel is a frequency and each
                column is a window. It has a size of (..., fft_size, n_frame, 2).
            length (Optional[int]): The amount to trim the signal by (i.e. the
                original signal length). (Default: whole signal)

        Returns:
            Tensor: Least squares estimation of the original signal of size (..., signal_length)
        """  # noqa E501
        if is_torch_1_8_plus and torch.is_complex(stft_matrix):
            stft_matrix = torch.view_as_real(stft_matrix)

        stft_matrix_dim = stft_matrix.dim()
        assert 3 <= stft_matrix_dim, "Incorrect stft dimension: %d" % (stft_matrix_dim)
        assert stft_matrix.numel() > 0

        if stft_matrix_dim == 3:
            # Add a channel dimension
            stft_matrix = stft_matrix.unsqueeze(0)

        # Pack batch
        shape = stft_matrix.size()
        stft_matrix = stft_matrix.view(-1, shape[-3], shape[-2], shape[-1])

        dtype, device = stft_matrix.dtype, stft_matrix.device
        fft_size = stft_matrix.size(1)
        assert (self.onesided and self.n_fft // 2 + 1 == fft_size) or (
            not self.onesided and self.n_fft == fft_size
        ), (
            "one_sided implies that n_fft // 2 + 1 == fft_size and "
            "not one_sided implies n_fft == fft_size. "
            + "Given values were onesided: %s, n_fft: %d, fft_size: %d"
            % ("True" if self.onesided else False, self.n_fft, fft_size)
        )

        # There must be overlap
        assert 0 < self.hop_length <= self.win_length
        assert 0 < self.win_length <= self.n_fft

        if self.dft_window is None:
            window = torch.ones(self.win_length, device=device, dtype=dtype)
            window = pad_center(window, self.n_fft)
        else:
            window = self.dft_window.to(dtype=dtype)

        # (channel, n_frame, fft_size, 2)
        stft_matrix = stft_matrix.transpose(1, 2)

        if self.impl == "fft":
            # (channel, n_frame, n_fft)
            istft_matrix = self._fft_istft(stft_matrix)
            # (channel, 1, n_samples)
            y = self._fold_istft_matrix(istft_matrix, window, length=length)
        elif self.impl == "matmul":
            istft_matrix = self._matmul_istft(stft_matrix)
            y = self._fold_istft_matrix(istft_matrix, window, length=length)
        elif self.impl == "conv":
            y = self._conv_istft(stft_matrix, window, length=length)

        # Unpack batch
        y = y.view(shape[:-3] + y.shape[-1:])

        if stft_matrix_dim == 3:  # Remove the channel dimension
            y = y.squeeze(0)

        return y

    def _fold_istft_matrix(self, istft_matrix, window, length=None):
        # istft_matrix: (channel, n_frame, n_fft)
        if self.normalized:
            if self.dft_window is None:
                istft_matrix /= self.normalized_scale
            else:
                scale = self.normalized_scale / self.n_fft
                istft_matrix *= scale
        else:
            istft_matrix /= self.n_fft

        assert istft_matrix.size(2) == self.n_fft
        n_frame = istft_matrix.size(1)

        # (channel, n_frame, n_fft)
        ytmp = istft_matrix * window.view(1, 1, self.n_fft)
        # (channel, n_fft, n_frame)
        # Each column of a channel is a frame which needs to be overlap added
        # at the right place.
        ytmp = ytmp.transpose(1, 2)

        # This does overlap add where the frames of ytmp are added such that
        # the i'th frame of ytmp is added starting at i*hop_length in the output.
        y = F.fold(
            ytmp,
            (1, (n_frame - 1) * self.hop_length + self.n_fft),
            (1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(2)

        # Overlap-add for the window function.
        # (1, n_fft, n_frame)
        window_sq = window.pow(2).view(self.n_fft, 1).repeat((1, n_frame)).unsqueeze(0)
        # (1, 1, expected_signal_len)
        window_envelop = F.fold(
            window_sq,
            (1, (n_frame - 1) * self.hop_length + self.n_fft),
            (1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(2)

        expected_signal_len = self.n_fft + self.hop_length * (n_frame - 1)
        assert y.size(2) == expected_signal_len
        assert window_envelop.size(2) == expected_signal_len

        half_n_fft = self.n_fft // 2
        # We need to trim the front padding away if center.
        start = half_n_fft if self.center else 0
        end = -half_n_fft if length is None else start + length

        y = y[:, :, start:end]
        window_envelop = window_envelop[:, :, start:end]

        # Check NOLA non-zero overlap condition.
        # window_envelop_lowest = window_envelop.abs().min()
        # assert window_envelop_lowest > 1e-11, "window overlap add min: %f" % (
        #     window_envelop_lowest
        # )
        #
        # (channel, expected_signal_len)
        # y = (y / window_envelop).squeeze(1)
        approx_nonzero_indices = window_envelop > tiny(window_envelop)
        y[:, :, approx_nonzero_indices[0, 0]] /= window_envelop[approx_nonzero_indices]
        y = y.squeeze(1)
        return y

    def _conv_stft(self, input, fft_window):
        """Conv1D based STFT implementation."""
        dtype = input.dtype
        has_batch_dim = input.dim() == 2

        # DFT basis (n_fft, N, 2)
        dft_bases = self.dft_bases[:, : self.cutoff].to(dtype=dtype)

        # (Batch, 1, nSamples)
        conv_input = input.unsqueeze(1) if has_batch_dim else input.view(1, 1, -1)
        # (2N, 1, n_fft)
        conv_kernel = (
            dft_bases.reshape(dft_bases.size(0), -1).transpose(0, 1).unsqueeze(1)
        )
        if fft_window is not None:
            conv_kernel *= fft_window.view(1, 1, -1)

        # (Batch, 2F, T)
        stft_matrix = F.conv1d(conv_input, conv_kernel, stride=self.hop_length)
        # (Batch, F, 2, T)
        stft_matrix = stft_matrix.view(
            stft_matrix.size(0), -1, 2, stft_matrix.size(-1)
        ).permute(0, 3, 1, 2)

        if not has_batch_dim:
            stft_matrix.squeeze_(0)
        return stft_matrix

    def _conv_istft(self, stft_matrix, window, length=None):
        """Conv1D based iSTFT implementation."""
        dtype = stft_matrix.dtype
        n_frame = stft_matrix.size(1)

        if self.onesided:
            other_half = stft_matrix[:, :, 1 : stft_matrix.size(2) - 1].flip((2,))
            other_half[..., 1] *= -1
            stft_matrix0 = torch.cat((stft_matrix, other_half), dim=2)
        else:
            stft_matrix0 = stft_matrix

        # DFT basis (n_fft, N, 2)
        dft_bases = self.dft_bases.to(dtype=dtype)

        # (Batch, T, F, 2) -> (Batch, 2F, T)
        conv_input = stft_matrix0.permute(0, 2, 3, 1).reshape(
            stft_matrix0.size(0), -1, stft_matrix0.size(1)
        )
        # (2F, 1, n_fft)
        conv_kernel = (
            dft_bases.reshape(dft_bases.size(0), -1).transpose(0, 1).unsqueeze(1)
        )
        conv_kernel *= window.view(1, 1, -1)

        # (Batch, 1, nSamples)
        y = F.conv_transpose1d(
            conv_input, conv_kernel, stride=self.hop_length, padding=0
        )

        if self.normalized:
            if self.dft_window is None:
                y /= self.normalized_scale
            else:
                scale = self.normalized_scale / self.n_fft
                y *= scale
        else:
            y /= self.n_fft

        # Ref: https://www.funcwj.cn/2020/04/10/conv1d-ops/
        win = torch.repeat_interleave(
            window.view(1, -1, 1), conv_input.size(-1), dim=-1
        )
        # (,_fft, 1, n_fft)
        I = torch.eye(window.shape[0], device=win.device, dtype=win.dtype).unsqueeze(1)
        # (1, 1, expected_signal_len)
        window_envelop = F.conv_transpose1d(
            win.pow(2), I, stride=self.hop_length, padding=0
        )

        expected_signal_len = self.n_fft + self.hop_length * (n_frame - 1)
        assert y.size(2) == expected_signal_len
        assert window_envelop.size(2) == expected_signal_len

        half_n_fft = self.n_fft // 2
        # We need to trim the front padding away if center.
        start = half_n_fft if self.center else 0
        end = -half_n_fft if length is None else start + length

        y = y[:, :, start:end]
        window_envelop = window_envelop[:, :, start:end]

        # Check NOLA non-zero overlap condition.
        # window_envelop_lowest = window_envelop.abs().min()
        # assert window_envelop_lowest > 1e-11, "window overlap add min: %f" % (
        #     window_envelop_lowest
        # )
        #
        # (channel, expected_signal_len)
        # y = (y / window_envelop).squeeze(1)
        approx_nonzero_indices = window_envelop > tiny(window_envelop)
        y[:, :, approx_nonzero_indices[0, 0]] /= window_envelop[approx_nonzero_indices]
        return y.squeeze(1)

    def _matmul_stft(self, input, fft_window):
        """Matrix multiplication based STFT implementation."""
        dtype = input.dtype
        has_batch_dim = input.dim() == 2

        # DFT basis (n_fft, N, 2)
        dft_bases = self.dft_bases[:, : self.cutoff].to(dtype=dtype)

        # Window the time series ([B,] n_fft, T)
        y_frames = frame(
            input, frame_length=self.n_fft, hop_length=self.hop_length, dim=-1
        )
        if fft_window is not None:
            y_frames = y_frames * fft_window

        # Compute the STFT matrix
        if has_batch_dim:
            stft_matrix = torch.einsum("nfa,bnt->btfa", dft_bases, y_frames)
        else:
            stft_matrix = torch.einsum("nfa,nt->tfa", dft_bases, y_frames)
        return stft_matrix

    def _matmul_istft(self, stft_matrix):
        """Matrix multiplication based iSTFT implementation."""
        dtype = stft_matrix.dtype

        # iSTFT is essentially `inv(dft_bases) @ stft_matrix`
        # DFT basis (n_fft, N, 2) -> (N, n_fft, 2)
        dft_bases_T = self.dft_bases.to(dtype=dtype).transpose(0, 1)
        if self.onesided:
            other_half = stft_matrix[:, :, 1 : stft_matrix.size(2) - 1].flip((2,))
            other_half[..., 1] *= -1
            stft_matrix0 = torch.cat((stft_matrix, other_half), dim=2)
        else:
            stft_matrix0 = stft_matrix
        istft_matrix = torch.einsum("nfa,btna->btf", dft_bases_T, stft_matrix0)
        return istft_matrix

    def _fft_stft(self, input, fft_window):
        """FFT based STFT implementation."""
        has_batch_dim = input.dim() == 2

        # Window the time series ([B,] n_fft, T)
        y_frames = frame(
            input, frame_length=self.n_fft, hop_length=self.hop_length, dim=-1
        )
        if fft_window is not None:
            y_frames = y_frames * fft_window

        # Compute the STFT matrix
        if has_batch_dim:
            if is_torch_1_8_plus:
                fft_ = torch.fft.rfft if self.onesided else torch.fft.fft
                stft_matrix = torch.view_as_real(
                    fft_(y_frames, dim=1, norm="backward")
                ).transpose(1, 2)
            else:
                stft_matrix = torch.rfft(
                    y_frames.transpose(-1, -2),
                    1,
                    normalized=False,
                    onesided=self.onesided,
                )
        else:
            if is_torch_1_8_plus:
                fft_ = torch.fft.rfft if self.onesided else torch.fft.fft
                stft_matrix = torch.view_as_real(
                    fft_(y_frames, dim=0, norm="backward")
                ).transpose(0, 1)
            else:
                stft_matrix = torch.rfft(
                    y_frames.transpose(-1, -2),
                    1,
                    normalized=False,
                    onesided=self.onesided,
                )
        return stft_matrix

    def _fft_istft(self, stft_matrix):
        """iFFT based iSTFT implementation."""
        if is_torch_1_8_plus:
            if not torch.is_complex(stft_matrix):
                stft_matrix = torch.view_as_complex(stft_matrix)
            if not self.onesided:
                stft_matrix = stft_matrix[:, :, : self.n_fft // 2 + 1]
            istft_matrix = torch.fft.irfft(
                stft_matrix, n=self.n_fft, dim=2, norm="forward"
            )
        else:
            istft_matrix = torch.irfft(
                stft_matrix,
                1,
                normalized=False,
                onesided=self.onesided,
                signal_sizes=(self.n_fft,),
            )
            istft_matrix *= self.n_fft
        # (channel, n_frame, n_fft)
        return istft_matrix

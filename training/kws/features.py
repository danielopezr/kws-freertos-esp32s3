"""Reference MFCC front end.

This is the single source of truth for feature extraction: the firmware (T2) must
reproduce these exact steps in C, so everything is written with plain numpy and
every constant is explicit.

Pipeline for one 1 s clip (16 000 samples, int16):
  1. int16 -> float32 in [-1, 1)            (divide by 32768)
  2. frames of 640 samples (40 ms), hop 320 (20 ms)  -> 49 frames
  3. periodic Hann window
  4. 1024-point real FFT, power spectrum |X|^2      -> 513 bins
  5. 40 triangular mel filters (HTK mel scale), 20 Hz - 4000 Hz
  6. natural log(mel_energy + 1e-6)
  7. orthonormal DCT-II, keep 40 coefficients       -> (49, 40)
"""

import numpy as np

SAMPLE_RATE = 16_000
CLIP_SAMPLES = 16_000
FRAME_LEN = 640
HOP = 320
N_FFT = 1024
N_MELS = 40
N_MFCC = 40
F_MIN = 20.0
F_MAX = 4_000.0
LOG_EPS = 1e-6
N_FRAMES = 1 + (CLIP_SAMPLES - FRAME_LEN) // HOP  # 49


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def hann_window(n=FRAME_LEN):
    """Periodic Hann window: 0.5 - 0.5 cos(2*pi*i/n)."""
    i = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * i / n)).astype(np.float32)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE, f_min=F_MIN, f_max=F_MAX):
    """Triangular filters over the n_fft//2+1 power-spectrum bins, shape (n_mels, n_bins)."""
    n_bins = n_fft // 2 + 1
    bin_hz = np.arange(n_bins, dtype=np.float64) * sr / n_fft
    edges_hz = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2))
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, center, hi = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]
        rising = (bin_hz - lo) / (center - lo)
        falling = (hi - bin_hz) / (hi - center)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb.astype(np.float32)


def dct_matrix(n_in=N_MELS, n_out=N_MFCC):
    """Orthonormal DCT-II as a matrix, shape (n_out, n_in)."""
    k = np.arange(n_out, dtype=np.float64)[:, None]
    n = np.arange(n_in, dtype=np.float64)[None, :]
    d = np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * n_in)) * np.sqrt(2.0 / n_in)
    d[0] /= np.sqrt(2.0)
    return d.astype(np.float32)


_WINDOW = hann_window()
_MEL_FB = mel_filterbank()
_DCT = dct_matrix()


def fit_length(audio):
    """Zero-pad or trim to exactly CLIP_SAMPLES (last axis)."""
    audio = np.asarray(audio)
    n = audio.shape[-1]
    if n >= CLIP_SAMPLES:
        return audio[..., :CLIP_SAMPLES]
    pad = [(0, 0)] * (audio.ndim - 1) + [(0, CLIP_SAMPLES - n)]
    return np.pad(audio, pad)


def mfcc(audio):
    """MFCC for one clip (n,) or a batch (b, n) of int16 or float audio.

    Returns float32 of shape (49, 40) or (b, 49, 40).
    """
    x = fit_length(audio)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    idx = np.arange(FRAME_LEN)[None, :] + HOP * np.arange(N_FRAMES)[:, None]  # (49, 640)
    frames = x[:, idx] * _WINDOW                                                # (b, 49, 640)
    spec = np.fft.rfft(frames, n=N_FFT, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)                # (b, 49, 513)
    mel = power @ _MEL_FB.T                                                     # (b, 49, 40)
    out = np.log(mel + LOG_EPS) @ _DCT.T                                        # (b, 49, 40)
    out = out.astype(np.float32)
    return out[0] if single else out


def mfcc_batched(clips, batch=1024):
    """MFCC for a large (N, n) int16 array without building huge intermediates."""
    out = np.empty((len(clips), N_FRAMES, N_MFCC), dtype=np.float32)
    for i in range(0, len(clips), batch):
        out[i:i + batch] = mfcc(clips[i:i + batch])
    return out

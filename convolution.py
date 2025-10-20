""" convolution.py
convolution.py - Convolution utilities for audio signals
Author: Petr Němec

This module provides functions for performing convolution operations
on 1D audio signals, including full, same-length, and circular convolution modes.
It also includes utilities for loading audio files, peak normalization,
and applying ADSR envelopes.

Functions:
- pair_convolution(sig, ker=None, mode="full"): Convolve two signals with specified mode.
- load_signal(path): Load an audio signal from a file.
- load_samples(folder_path, normalize=False): Load all .wav files from a folder.
- process_self(...): Perform self-convolution on audio files in a folder.
- process_pairs(...): Convolve audio files with specified kernels.

Usage examples are provided in the __main__ section.
"""
import os
import numpy as np
import soundfile as sf

# ---------- utilities ----------
def _next_fast_len_ge(n):
    """Return the next fast length >= n for FFT.
    
    Parameters
    ----------
    n : int, minimum length
    
    Returns
    -------
    m : int, next fast length >= n
    """
    try:
        from numpy.fft import next_fast_len
        return next_fast_len(n)
    except Exception:
        p = 1
        while p < n: p <<= 1
        return p

def _center_slice(y_full, N):
    """Return center N samples from y_full (or zero-pad if N > len(y_full)).
    
    Parameters
    ----------
    y_full : array [L], full-length signal
    N      : int, desired length

    Returns
    -------
    out : array [N], centered slice or zero-padded
    """
    L = len(y_full)
    if N >= L:
        out = np.zeros(N, dtype=y_full.dtype)
        start = (N - L)//2
        out[start:start+L] = y_full
        return out
    start = (L - N)//2
    return y_full[start:start+N]

def peak_normalize(x, peak=0.99):
    """Peak-normalize signal x to given peak value.
    Parameters
    ----------
    x    : array [N] or [N, C]
    peak : float, desired peak value
    Returns
    -------
    y : array [N] or [N, C]
    """
    m = np.max(np.abs(x))
    return (peak/m)*x if m > 0 else x

def apply_adsr_envelope(x, fs, attack=0.2, decay=0.1, sustain_level=0.9, release=0.3):
    """Apply a simple ADSR envelope to signal x.
    Parameters
    ----------
    x             : array [N] or [N, C]
    fs            : int, sample rate
    attack        : float, attack time in seconds
    decay         : float, decay time in seconds
    sustain_level : float, sustain level (0 to 1)
    release       : float, release time in seconds
    Returns
    -------
    y : array [N] or [N, C]
    """
    N = x.shape[0]
    env = np.ones(N, dtype=np.float64)

    a_samples = int(attack * fs)
    d_samples = int(decay * fs)
    r_samples = int(release * fs)
    s_start = a_samples + d_samples
    r_start = N - r_samples

    # Attack
    if a_samples > 0:
        env[:a_samples] = np.linspace(0, 1, a_samples)
    # Decay
    if d_samples > 0:
        env[a_samples:s_start] = np.linspace(1, sustain_level, d_samples)
    # Sustain
    env[s_start:r_start] = sustain_level
    # Release
    if r_samples > 0:
        env[r_start:] = np.linspace(sustain_level, 0, r_samples)

    return x * env[:, None] if x.ndim == 2 else x * env
# ---------- core 1D engines ----------
def _pair_conv_1d_linear_full(x, h):
    """Full linear convolution of two 1D real signals (length = Nx+Nh-1)."""
    Nx, Nh = len(x), len(h)
    out_len = Nx + Nh - 1
    fft_len = _next_fast_len_ge(out_len)
    X = np.fft.rfft(x, n=fft_len)
    H = np.fft.rfft(h, n=fft_len)
    Y = X * H                      # keep complex phase
    y = np.fft.irfft(Y, n=fft_len)[:out_len]
    return y

def _pair_conv_1d_circular(x, h, ref_len=None):
    """
    Circular convolution, length = ref_len (default len(x)).
    Note: circular conv is modulo-N wrap-around.
    """
    N = len(x) if ref_len is None else ref_len
    X = np.fft.rfft(x, n=N)
    H = np.fft.rfft(h, n=N)
    Y = X * H
    y = np.fft.irfft(Y, n=N)
    return y

# ---------- public API ----------
def pair_convolution(sig, ker=None, mode="full"):
    """
    Convolution of two signals with length behavior controlled by `mode`.
    If `ker is None`, performs self-convolution of `sig`.

    Parameters
    ----------
    sig  : array [N] or [N, C]   (reference for 'same-*' length)
    ker  : array [M] or [M, C] or None
    mode : "full" | "same-first" | "same-center" | "circular"

    Returns
    -------
    y : array [L] or [L, C]
    """
    if ker is None:
        ker = sig

    # Ensure 2D [N,C]
    if sig.ndim == 1: sig = sig[:, None]
    if ker.ndim == 1: ker = ker[:, None]

    Nx, Cx = sig.shape
    Nh, Ch = ker.shape

    # Channel strategy:
    # - if Cx == Ch: convolve per-channel
    # - else: downmix both to mono and return mono
    if Cx == Ch:
        outs, maxL = [], 0
        for c in range(Cx):
            x = sig[:, c]
            h = ker[:, c]
            if mode == "circular":
                y = _pair_conv_1d_circular(x, h, ref_len=Nx)
            else:
                y_full = _pair_conv_1d_linear_full(x, h)
                if mode == "full":
                    y = y_full
                elif mode == "same-first":
                    y = y_full[:Nx]
                elif mode == "same-center":
                    y = _center_slice(y_full, Nx)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            outs.append(y); maxL = max(maxL, len(y))
        Y = np.zeros((maxL, Cx), dtype=np.float64)
        for c, y in enumerate(outs):
            Y[:len(y), c] = y
        return Y
    else:
        # Downmix both to mono (mean) if channel counts differ
        x_mono = np.mean(sig, axis=1)
        h_mono = np.mean(ker, axis=1)
        if mode == "circular":
            y = _pair_conv_1d_circular(x_mono, h_mono, ref_len=Nx)
        else:
            y_full = _pair_conv_1d_linear_full(x_mono, h_mono)
            if mode == "full":
                y = y_full
            elif mode == "same-first":
                y = y_full[:Nx]
            elif mode == "same-center":
                y = _center_slice(y_full, Nx)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        return y[:, None]  # mono

# ---------- I/O helpers ----------
def load_signal(path):
    """Load audio signal from file.
    Parameter
    ---------
    path : str, file path

    Returns
    ------- 
    x  : array [N,C]
    fs : int, sample rate
    """
    x, fs = sf.read(path, always_2d=True)  # [N,C]
    return x, fs

def load_samples(folder_path, normalize=False):
    """Load all .wav files from folder_path.
    Parameters
    ----------
    folder_path : str, path to folder with .wav files
    normalize   : bool, whether to peak-normalize each signal
    
    Returns
    -------
    signals : list of arrays [N,C]
    fs      : int, sample rate (assumed same for all files)
    names   : list of str, filenames
    """
    signals, names, fs = [], [], None
    for fn in sorted(os.listdir(folder_path)):
        if fn.lower().endswith('.wav'):
            x, fs = load_signal(os.path.join(folder_path, fn))
            if normalize:
                x = peak_normalize(x, 0.99)
            signals.append(x); names.append(fn)
    return signals, fs, names

# ---------- example pipelines ----------
def process_self(inputs_folder='samples', outputs_folder='out_self', mode="full", n=2, normalize=False, adsr=False):
    """Self-convolution n-fold (uses pair_convolution repeatedly if n>2).
    
    Parameters
    ----------
    inputs_folder  : str, folder with input .wav files
    outputs_folder : str, folder to write output .wav files
    mode           : str, convolution mode ("full", "same-first", "same-center", "circular")
    n              : int, number of self-convolutions
    normalize      : bool, whether to peak-normalize input signals
    adsr           : bool, whether to apply ADSR envelope to output signals
    """
    os.makedirs(outputs_folder, exist_ok=True)
    signals, fs, names = load_samples(inputs_folder, normalize=normalize)
    for sig, name in zip(signals, names):
        # n-fold via repeated pair_convolution(sig, None) -> equivalent to X**n
        y = sig
        for _ in range(n-1):
            y = pair_convolution(y, ker=sig, mode=mode)
        y = peak_normalize(y, 0.99)
        y = apply_adsr_envelope(y, fs) if adsr else y
        out = os.path.join(outputs_folder, f"{os.path.splitext(name)[0]}_self_{mode}_{n}.wav")
        sf.write(out, y, fs)
        print(f"Wrote: {out}  {y.shape} fs={fs}")

def process_pairs(inputs_folder='samples', kernel_path=None, kernels_folder=None,
                  outputs_folder='out_pair', mode="full", normalize=False, adsr=False):
    """
    Convolve each file in inputs_folder with:
      - a single kernel file (kernel_path), OR
      - a kernel with the SAME FILENAME in kernels_folder, if provided.

    Parameters
    ----------  
    inputs_folder  : str, folder with input .wav files
    kernel_path    : str or None, path to single kernel .wav file
    kernels_folder : str or None, folder with kernel .wav files (matched by filename)
    outputs_folder : str, folder to write output .wav
    mode           : str, convolution mode ("full", "same-first", "same-center", "circular")
    normalize      : bool, whether to peak-normalize input signals
    adsr           : bool, whether to apply ADSR envelope to output signals
    """
    os.makedirs(outputs_folder, exist_ok=True)
    signals, fs, names = load_samples(inputs_folder, normalize=normalize)

    # Load a single shared kernel if provided
    shared_kernel = None
    if kernel_path:
        shared_kernel, fs_k = load_signal(kernel_path)
        if fs_k != fs:
            raise ValueError("Sample rates differ between input and kernel.")

    for sig, name in zip(signals, names):
        if shared_kernel is not None:
            ker = shared_kernel
            tag = os.path.splitext(os.path.basename(kernel_path))[0]
        elif kernels_folder:
            ker_file = os.path.join(kernels_folder, name)  # match by filename
            if not os.path.exists(ker_file):
                print(f"[warn] kernel not found for {name}; skipping")
                continue
            ker, fs_k = load_signal(ker_file)
            if fs_k != fs:
                raise ValueError(f"Sample rates differ for {name}.")
            tag = os.path.splitext(name)[0]
        else:
            raise ValueError("Provide either kernel_path or kernels_folder.")

        y = pair_convolution(sig, ker=ker, mode=mode)
        y = peak_normalize(y, 0.99)
        y = apply_adsr_envelope(y, fs) if adsr else y
        out_name = f"{os.path.splitext(name)[0]}__with__{tag}_{mode}.wav"
        out_path = os.path.join(outputs_folder, out_name)
        sf.write(out_path, y, fs)
        print(f"Wrote: {out_path}  {y.shape} fs={fs}")

# ---------- main examples ----------
if __name__ == "__main__":
    # Uncomment to run examples
    # process_self(inputs_folder='samples', outputs_folder='conv', mode="full", n=3)
    # process_self(inputs_folder='samples', outputs_folder='conv', mode="same-first", n=3)
    # process_self(inputs_folder='samples', outputs_folder='conv', mode="same-center", n=4)
    # process_self(inputs_folder='samples', outputs_folder='self_conv',  mode="circular", n=3, normalize=True, adsr=True)

    # process_pairs(inputs_folder='samples', kernel_path='kernels/piano-001.wav', mode="full", normalize=True, adsr=True)
    # process_pairs(inputs_folder='samples', kernel_path='kernels/piano-002.wav', mode="full")

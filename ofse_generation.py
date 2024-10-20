import numpy as np
from scipy.fft import fft, fftfreq
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
from python_speech_features import sigproc
import matplotlib.pyplot as plt
import librosa
from scipy.signal.windows import hamming




def getansifrequencies(fraction, limits=None):
    """ ANSI s1.11-2004 && IEC 61260-1-2014
    Array of frequencies and its edges according to the ANSI and IEC standard.

    :param fraction: Bandwidth 'b'. Examples: 1/3-octave b=3, 1-octave b=1,
    2/3-octave b = 3/2
    :param limits: It is a list with the minimum and maximum frequency that
    the array should have.
    :returns: Frequency array, lower edge array and upper edge array
    :rtype: list, list, list
    """
    if limits is None:
        limits = [12, 40000]

    # Octave ratio g (ANSI s1.11, 3.2, pg. 2)
    g = 10 ** (3 / 10)  # Or g = 2
    # Reference frequency (ANSI s1.11, 3.4, pg. 2)
    fr = 1000

    # Get starting index 'x' and first center frequency
    x = _initindex(limits[0], fr, g, fraction)
    freq = _ratio(g, x, fraction) * fr

    # Get each frequency until reach maximum frequency
    freq_x = 0
    while freq_x * _bandedge(g, fraction) < limits[1]:
        # Increase index
        x = x + 1
        # New frequency
        freq_x = _ratio(g, x, fraction) * fr
        # Store new frequency
        freq = np.append(freq, freq_x)

    # Get band-edges
    freq_d = freq / _bandedge(g, fraction)
    freq_u = freq * _bandedge(g, fraction)

    return freq.tolist(), freq_d.tolist(), freq_u.tolist()

def _initindex(f, fr, g, b):
    if b % 2:  # ODD ('x' solve from ANSI s1.11, eq. 3)
        return np.round(
                (b * np.log(f / fr) + 30 * np.log(g)) / np.log(g)
                )
    else:  # EVEN ('x' solve from ANSI s1.11, eq. 4)
        return np.round(
                (2 * b * np.log(f / fr) + 59 * np.log(g)) / (2 * np.log(g))
                )
    
def _ratio(g, x, b):
    if b % 2:  # ODD (ANSI s1.11, eq. 3)
        return g ** ((x - 30) / b)
    else:  # EVEN (ANSI s1.11, eq. 4)
        return g ** ((2 * x - 59) / (2 * b))
    
def _bandedge(g, b):
    # Band-edge ratio (ANSI s1.11, 3.7, pg. 3)
    return g ** (1 / (2 * b))


def split_audio(audio_data, sample_rate, segment_size=0.5):
    segment_size_samples = int(segment_size * sample_rate)
    segments = [audio_data[i:i + segment_size_samples] for i in range(0, len(audio_data), segment_size_samples) if i + segment_size_samples <= len(audio_data)]
    return segments

def apply_octave_filter_to_power_spectrum(x, fs, fraction=1, limits=None):
    if limits is None:
        limits = (12, 40000)
    
    # Perform FFT and calculate power spectrum
    X_fft = fft(x)
    power_spectrum = np.abs(X_fft) ** 2
    
    n = len(x)
    freq = fftfreq(n, d=1/fs)
    
    # Initialize filtered power spectrum
    filtered_power_spectrum = np.zeros_like(power_spectrum)
    
    # Generate frequency bands
    center_frequencies, freq_d, freq_u = getansifrequencies(fraction, limits)
    
    band_energies = []
    for lower, upper in zip(freq_d, freq_u):
        band_mask = (freq >= lower) & (freq <= upper)
        band_energy = np.sum(power_spectrum[band_mask])
        band_energies.append(band_energy)
    
    # Return band energies directly, without splitting the return values
    return center_frequencies, band_energies

def compute_octave_spectrum_energey(segment, sample_rate, winlen=0.02, winstep=0.01, fraction=3):
    frame_len = int(winlen * sample_rate)
    step_len = int(winstep * sample_rate)
    frames = [segment[i:i+frame_len] for i in range(0, len(segment)-frame_len+1, step_len)]
    frames = np.array([frame * hamming(len(frame)) for frame in frames])
    
    # Initialize a list to hold the band energies for all frames
    all_band_energies = []
    
    # Loop through each frame and apply octave filtering
    for frame in frames:
        # Correctly receive the returned values from the function
        center_frequencies, frame_band_energies = apply_octave_filter_to_power_spectrum(frame, sample_rate, fraction=fraction)
        all_band_energies.append(frame_band_energies)
    
    all_band_energies = np.array(all_band_energies)
    
    # Log-transform the band energies
    log_band_energies = 10 * np.log10(all_band_energies + np.finfo(float).eps)
    
    return log_band_energies

file_path = r'enter/file/path'
audio_data, sr = librosa.load(file_path, sr=44100)
segments = split_audio(audio_data, sr)

# OFSE calculation for each segment
ofse_results = [compute_octave_spectrum_energey(segment, sr) for segment in segments]
ofse_results = np.array(ofse_results)

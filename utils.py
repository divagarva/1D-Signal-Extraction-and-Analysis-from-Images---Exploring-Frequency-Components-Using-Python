# utils.py
import numpy as np
from scipy.fft import fft, fftfreq

def extract_1d_signal(image, axis='row'):
    """
    Extract a 1D signal from the image.
    :param image: Grayscale image as a 2D array.
    :param axis: 'row' to extract from the middle row, 'column' to extract from the middle column.
    :return: 1D array of pixel values.
    """
    if axis == 'row':
        # Extract the middle row
        row_index = image.shape[0] // 2
        return image[row_index, :]
    elif axis == 'column':
        # Extract the middle column
        col_index = image.shape[1] // 2
        return image[:, col_index]
    else:
        print("Invalid axis specified. Use 'row' or 'column'.")
        return np.array([])

def apply_signal_analysis(signal, analysis_type='fft'):
    """
    Apply analysis techniques to the 1D signal.
    :param signal: 1D array of pixel values.
    :param analysis_type: Type of analysis ('fft' for Fourier Transform).
    :return: Analyzed signal data.
    """
    if analysis_type == 'fft':
        # Apply Fourier Transform to the 1D signal
        signal_fft = fft(signal)
        magnitude_spectrum = np.abs(signal_fft)
        return magnitude_spectrum
    else:
        print("Unsupported analysis type.")
        return signal

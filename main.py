# main.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import extract_1d_signal, apply_signal_analysis

def main():
    # Load an image
    image_path = 'tiger.jpg'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Unable to load image.")
        return

    # Extract a 1D signal (e.g., middle row)
    row_signal = extract_1d_signal(image, axis='row')

    # Plot the original image and the extracted signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.plot(row_signal, color='blue')
    plt.title('Extracted 1D Signal (Middle Row)')
    plt.xlabel('Pixel Index')
    plt.ylabel('Intensity')
    plt.grid(True)

    # Apply signal analysis (e.g., Fourier Transform)
    analyzed_signal = apply_signal_analysis(row_signal, analysis_type='fft')

    # Plot the analyzed signal
    plt.figure(figsize=(12, 6))
    plt.plot(analyzed_signal, color='red')
    plt.title('Analyzed Signal - Frequency Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

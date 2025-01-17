import os
import numpy as np
from scipy.io import wavfile
import librosa
from sklearn.model_selection import train_test_split
from collections import Counter


def store_recording(directory, wav_file):
    file_path = os.path.join(directory, wav_file)

    try:
        _, data = wavfile.read(file_path)

        label = ord(wav_file.split('_')[0]) - 65
        return data, label

    except Exception as e:
        print(f"Error loading file {wav_file}: {e}")


def load_wav_files(directory):
    dataset = []
    labels = []

    for wav_file in os.listdir(directory):
        data, label = store_recording(directory, wav_file)
        dataset.append(data)
        labels.append(label)

    return np.array(dataset), np.array(labels)


def load_power_recordings(directory):
    dataset = []
    labels = []

    for wav_file in os.listdir(directory):
        if wav_file[2] != 'P':
            continue
        data, label = store_recording(directory, wav_file)
        dataset.append(data)
        labels.append(label)

    return np.array(dataset), np.array(labels)


def load_audio_recordings(directory):
    dataset = []
    labels = []

    for wav_file in os.listdir(directory):
        if wav_file[2] != 'A':
            continue
        data, label = store_recording(directory, wav_file)
        dataset.append(data)
        labels.append(label)

    return np.array(dataset), np.array(labels)


def min_max_normalization(dataset):
    """
    Apply min-max normalization to the dataset.
    Considering that wav file has bitrate 16kpbs, meaning max value of 32767

    Parameters
    ---------------
        dataset: numpy.ndarray
            The dataset to be normalized.

    Returns
    ---------------
        normalized_data: numpy.ndarray:
            The normalized dataset.
    """
    normalized_data = []
    for i in range(len(dataset)):
        audio = dataset[i]
        # temp = audio.astype(np.float32) / 32767.0
        normalized_data.append(audio.astype(np.float32) / 32767.0)

    return np.array(normalized_data)


def z_score_normalization(dataset):
    """
    Apply mean-and-variance normalization (z-score normalization) to the dataset.

    Parameters
    ---------------
        dataset: numpy.ndarray
            The dataset to be normalized.

    Returns
    ---------------
        normalized_data: numpy.ndarray:
            The normalized dataset.
    """
    mean_val = np.mean(dataset)
    std_val = np.std(dataset)
    normalized_data = (dataset - mean_val) / std_val
    return normalized_data


def pcen_normalization(dataset, sr=22050, hop_length=512, gain=0.98, power=0.5, time_constant=0.4, eps=1e-6):
    """
    Apply Per-Channel Energy Normalization (PCEN) to the dataset.

    Parameters:
        dataset (numpy.ndarray): The dataset to be normalized (audio samples).
        sr (int, optional): The sample rate of the audio data. Default is 22050.
        hop_length (int, optional): The number of samples between successive frames in the audio data.
                                    Default is 512.
        gain (float, optional): The gain factor for PCEN. Default is 0.98.
        power (float, optional): The exponent for calculating the energy in PCEN. Default is 0.5.
        time_constant (float, optional): The time constant for the adaptive gain control in PCEN.
                                         Default is 0.4.
        eps (float, optional): A small constant to avoid division by zero. Default is 1e-6.

    Returns:
        numpy.ndarray: The PCEN-normalized dataset.
    """
    # Convert the data to floating-point if it's in integer format
    if dataset.dtype == np.int16:
        dataset = dataset.astype(np.float32) / 32767.0  # Assuming 16-bit audio (range [-32768, 32767])

    # Calculate the squared magnitude spectrogram using STFT from librosa
    spectrogram = np.abs(librosa.stft(dataset, hop_length=hop_length)) ** 2

    # Apply power compression to the spectrogram
    compressed_spectrogram = np.power(spectrogram, power)

    # Apply adaptive gain control using a time-domain IIR filter
    alpha = 1.0 / (sr * time_constant)
    smoothed_spectrogram = np.zeros_like(compressed_spectrogram)
    for t in range(1, compressed_spectrogram.shape[1]):
        smoothed_spectrogram[:, t] = gain * smoothed_spectrogram[:, t - 1] + (1 - gain) * compressed_spectrogram[:,
                                                                                          t - 1]

    # Calculate the PCEN
    pcen = compressed_spectrogram / (eps + smoothed_spectrogram) ** power

    # Inverse Short-Time Fourier Transform (ISTFT) to obtain the PCEN-normalized audio samples
    normalized_data = librosa.istft(np.sqrt(pcen), hop_length=hop_length)

    return normalized_data


def one_vs_all(X, y, label):
    label = ord(label) - 65

    class_counts = Counter(y)

    label_class_size = class_counts[label]

    # Create arrays to store the new balanced dataset
    X_balanced = []
    y_balanced = []

    # Keep all instances of label class
    label_class_indicies = np.where(y == label)[0]
    X_balanced.extend(X[label_class_indicies])
    y_balanced.extend(y[label_class_indicies])

    # Keep an equal amount of instances from each remaining class
    rest_class_size = label_class_size // 8
    for class_label, count in class_counts.items():
        if class_label != label:
            indices = np.where(y == class_label)[0]
            selected_indices = np.random.choice(indices, size=rest_class_size, replace=False)
            X_balanced.extend(X[selected_indices])
            y_balanced.extend(y[selected_indices])

    # Change labels to binary classification problem
    y_balanced = np.array(y_balanced)
    y_balanced = np.where(y_balanced == label, 0, 1)

    # Convert balanced lists to numpy arrays
    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)

    return X_balanced, y_balanced


if __name__ == '__main__':
    # Load the files from the folder with the separated recordings
    directory = 'databases/separated_16_8'  # Change to correct directory
    # X, y = load_wav_files(directory)
    # X, y = load_audio_recordings(directory)
    X, y = load_power_recordings(directory)

    # Normalize the dataset
    # X = pcen_normalization(X)
    X = min_max_normalization(X)
    # X = z_score_normalization(X)
    
    # np.savez("saves/autoencoder/full.npz", X_data=X)

    for grid in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        print(f'Grid {grid} processing...', end='')

        X_grid, y_grid = one_vs_all(X, y, grid)

        # Split the dataset and labels into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_grid, y_grid, test_size=0.2, random_state=42)

        # Save the dataset
        np.savez(f'saves/power/train_{grid}_vs_all.npz', X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
        print("done")

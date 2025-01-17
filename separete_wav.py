import os
from scipy.io import wavfile


def separate_wav_file(wav_file_path, duration, overlap, output_folder, part_name='part'):
    """
    Separates a wav file into smaller parts and save them in a folder

    Parameters
    ---------------
    duration: int
        The duration of each part (in seconds)
    overlap:  int
        How often start a new part (in seconds)
    wav_file_path: str

    output_folder: str
        The name of the output folder. If there is no such folder, it creates it
    part_name: str, optional
        The name of the output file-part, before the count (1,2,...) (default = part_XX)

    """
    # Check if the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the WAV file
    fs, data = wavfile.read(wav_file_path)

    # Convert M and F from seconds to nof_samples
    duration_samples = duration * fs
    overlap_samples = overlap * fs

    start_sample = 0
    end_sample = duration_samples
    i = 0

    while end_sample < len(data):
        # Extract the part from the audio
        part = data[start_sample:end_sample - 1]

        # Save the part as a new WAV file
        part_filename = os.path.join(output_folder, f'{part_name}_{i + 1:03}.wav')
        wavfile.write(part_filename, fs, part)

        start_sample += overlap_samples
        end_sample += overlap_samples
        i += 1

    # Create the last part containing the last minutes of the audio
    if start_sample < len(data):
        end_sample = len(data)
        start_sample = end_sample - duration_samples

        # Extract the part from the audio
        part = data[start_sample:end_sample - 1]

        # Save the part as a new WAV file
        part_filename = os.path.join(output_folder, f'{part_name}_{i + 1:03}.wav')
        wavfile.write(part_filename, fs, part)


def get_recording_code(name):
    # Remove the ".wav" extension
    filename_without_extension = name[:-4]

    # Extract grid and counting from the filename
    XYY = filename_without_extension.split('_')[2:]
    return '_'.join(XYY)


def separate_database(directory, output_directory):
    """
    Function to iterate through all the wav files in a database

    :param directory:
    :return:
    """
    for grid in os.listdir(directory):
        grid_path = os.path.join(directory, grid)
        if os.path.isdir(grid_path):
            for subfolder in os.listdir(grid_path):
                subfolder_path = os.path.join(grid_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for recording in os.listdir(subfolder_path):
                        if recording.endswith('.wav'):
                            recording_path = os.path.join(subfolder_path, recording)
                            print(recording_path)

                            code = get_recording_code(recording)
                            separate_wav_file(recording_path, 16, 8, output_directory, part_name=code)
    print('Separation and saving completed.')


separate_database('databases/database_raw', 'databases/separated_16_8')

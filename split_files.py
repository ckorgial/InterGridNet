from pydub import AudioSegment
import math
import os


def split_audio_by_duration(input_file, output_dir, duration_minutes):
    name = os.path.basename(input_file).split('.')
    print(name[0])

    audio = AudioSegment.from_file(input_file)
    duration_ms = duration_minutes * 60 * 1000
    num_parts = math.ceil(len(audio) / duration_ms)

    for i in range(num_parts):
        start_time = i * duration_ms
        end_time = start_time + duration_ms
        part = audio[start_time:end_time]
        output_file = f"{output_dir}/{name[0]}_{i + 1}.{name[1]}"  # Output file format can be changed as per your requirement
        part.export(output_file, format="wav")


def process_dataset(dataset_dir, output_dir, duration_minutes):
    for grid_folder in os.listdir(dataset_dir):
        grid_folder_path = os.path.join(dataset_dir, grid_folder)
        if not os.path.isdir(grid_folder_path):
            continue

        audio_folder_path = os.path.join(grid_folder_path, "Audio_recordings")
        power_folder_path = os.path.join(grid_folder_path, "Power_recordings")
        if not os.path.isdir(audio_folder_path) or not os.path.isdir(power_folder_path):
            continue

        for file in os.listdir(audio_folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(audio_folder_path, file)
                split_audio_by_duration(file_path, output_dir, duration_minutes)

        for file in os.listdir(power_folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(power_folder_path, file)
                split_audio_by_duration(file_path, output_dir, duration_minutes)

# Usage example
dataset_dir = "Testing_dataset"  # Replace with the path to your dataset directory
new_dataset_dir = "test_2"
duration_minutes = 2  # Replace with the desired duration in minutes

#process_dataset(dataset_dir, new_dataset_dir, duration_minutes)
for file in os.listdir(dataset_dir):
    if file.endswith(".wav"):
        file_path = os.path.join(dataset_dir, file)
        split_audio_by_duration(file_path, new_dataset_dir, duration_minutes)



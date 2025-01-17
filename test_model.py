from scipy.io import wavfile
import numpy as np
from keras.models import load_model
import os
from sklearn.metrics import confusion_matrix

practice_ground_truth = 'AHCFFBGINDAFBDCINNAEHBBADCGNGBDDCHGEAIHIEHECFFNGEI'
test_ground_truth = 'NDDCDNNDAFANGBGBFCEHGHHHGHFDAIDNFHIIECBDENIBEFGNAGIINIGHAEFCCCFDGCECGIEICENBEEHADIHCGAABIHCNDBAGBFBB'

def majority_vote(chars_array):
    char_counts = {}  # Dictionary to store character counts

    # Count occurrences of each character in the array
    for char in chars_array:
        char_counts[char] = char_counts.get(char, 0) + 1

    # Find the character with the highest count (majority vote)
    majority_char = max(char_counts, key=char_counts.get)

    return majority_char


test_directory = 'databases/Testing_dataset/'  # Change path

# Load ML model
model_path = 'models/audio_classifier_model.h5'
model = load_model(model_path)

file_list = os.listdir(test_directory)
testing_samples = [file for file in file_list if file.lower().endswith('.wav')]

y_true = []
y_pred = []
for testing_sample in testing_samples:
    # Load one testing sample
    test_sample_path = test_directory + testing_sample
    sample_rate, audio_data = wavfile.read(test_sample_path)

    # Extract label
    temp = testing_sample.split('_')[1]
    label_index = int(temp.split('.')[0])
    label = test_ground_truth[label_index-1]

    # Split the sample in parts of 16s
    i = 0
    predictions = []
    while i+15999 < len(audio_data):
        testing_part = audio_data[i:i+15999]
        testing_part = np.expand_dims(testing_part, axis=-1)
        testing_part = testing_part[np.newaxis, :, :]

        # Normalize
        testing_part = testing_part.astype(np.float32) / 32767.0

        # Predict
        prediction = model.predict(testing_part)
        predicted_label = chr(np.argmax(prediction) + 65)
        predictions.append(predicted_label)

        # Next part
        i += 10000

    # Majority voting
    pred = majority_vote(predictions)
    print(f"Sample: {testing_sample}, Label: {label}, Prediction: {pred}")
    y_pred.append(pred)
    y_true.append(label)

conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


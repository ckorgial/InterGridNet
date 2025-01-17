from keras.models import load_model
import os
import numpy as np
from scipy.io import wavfile

model = load_model('models/FreqENF.h5')
AUDIO_LENGTH = 120000

folder_path = 'Practice_dataset'
wav_files = os.listdir(folder_path)

predictions = np.zeros(100)
predictionsX = []
for file in wav_files:
    print(file)
    file_path = os.path.join(folder_path, file)
    _, wav_data = wavfile.read(file_path)

    num_parts = len(wav_data) // AUDIO_LENGTH
    parts = np.array_split(wav_data, num_parts)

    file_predictions = []
    for part in parts:
        part = np.expand_dims(part, axis=0)
        prediction = model.predict(part)

        if prediction[0][0] == 1:
            file_predictions.append(50)
        else:
            file_predictions.append(60)

    predictionsX.append(file_predictions)
    majority_prediction = np.argmax(np.bincount(file_predictions))

    code = int(file.split('_')[1].split('.')[0])
    predictions[code-1] = majority_prediction

print(predictions)

#test_ground_truth = 'NDDCDNNDAFANGBGBFCEHGHHHGHFDAIDNFHIIECBDENIBEFGNAGIINIGHAEFCCCFDGCECGIEICENBEEHADIHCGAABIHCNDBAGBFBB'
test_ground_truth = 'AHCFFBGINDAFBDCINNAEHBBADCGNGBDDCHGEAIHIEHECFFNGEI'

# Create the NumPy array based on the conditions
array = np.array([0 if char == 'N' else 60 if char in ['A', 'C', 'I'] else 50 for char in test_ground_truth])

print(array)

count = 0
for i in range(len(array)):
    if predictions[i] != array[i]:
        print(test_ground_truth[i], predictions[i], predictionsX[i])

        if test_ground_truth[i] != 'N':
            count += 1

print(count)




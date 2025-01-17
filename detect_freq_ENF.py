import os
import numpy as np
import soundfile as sf
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


# Path to the folder containing the files
folder_path = 'database_split_2'
AUDIO_LENGTH = 120000

# CREATE TRAINING DATA & LABELS
ENF_freq = {'A': 60, 'B': 50, 'C': 60, 'D': 50, 'E': 50, 'F': 50, 'G': 50, 'H': 50, 'I': 60}

train_data = np.zeros([len(os.listdir(folder_path)), AUDIO_LENGTH])
labels = np.zeros(len(os.listdir(folder_path)))
# Iterate through each file in the folder
for i, filename in enumerate(os.listdir(folder_path)):
    file_path = os.path.join(folder_path, filename)

    # Check if the path is a file and if it has a .wav extension
    if os.path.isfile(file_path) and filename.lower().endswith('.wav'):
        try:
            # Read the WAV file
            audio_data, _ = sf.read(file_path)
            train_data[i] = audio_data

            # Find label from the Grid name
            labels[i] = ENF_freq[filename.split('_')[2]]

        except Exception as e:
            print(f"Error reading file: {filename} - {e}")


# Convert labels to one-hot encoding
labels = pd.get_dummies(labels).astype('int').values

# # Normalize input data (optional but often beneficial for model training)
# train_data = train_data / np.max(train_data)

# Reshape train_data to match the input shape of the CNN (816, AUDIO_LENGTH, 1)
train_data = np.expand_dims(train_data, axis=-1)

# Create the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(AUDIO_LENGTH, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for epoch in range(10):
    # Shuffle the data
    perm = np.random.permutation(len(labels))
    train_data = train_data[perm]
    labels = labels[perm]

    # Split the data into training and testing sets (80% for training, 20% for testing)
    split_index = int(0.85 * len(train_data))
    x_train = train_data[:split_index]
    y_train = labels[:split_index]
    x_test = train_data[split_index:]
    y_test = labels[split_index:]

    # Train the model
    model.fit(x_train, y_train, batch_size=16, validation_data=(x_test, y_test))

# Evaluate the model
'''loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")'''

model.save('models/FreqENF.h5')

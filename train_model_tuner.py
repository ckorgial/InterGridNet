import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, BatchNormalization, LeakyReLU, Permute, GRU, Input, GlobalAveragePooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel, RandomSearch

# Enable dynamic memory allocation for the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Num GPUs Available: {len(physical_devices)}")

class DynamicAudioCNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()

        # Input layer
        model.add(Input(shape=self.input_shape))

        # Example hyperparameters for dynamic layers
        num_conv_layers = hp.Int('num_conv_layers', min_value=3, max_value=5)
        filters = hp.Int('conv_filters', min_value=128, max_value=256, step=128)
        #kernel_size = hp.Int('kernel_size', min_value=3, max_value=7, step=2)
        units = hp.Int('gru_units', min_value=512, max_value=1024, step=512)
        dense_units = hp.Int('dense_units', min_value=64, max_value=512, step=128)
        #dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
        #learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')

        # Initial convolutional layer
        model.add(Conv1D(filters=filters, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))

        # Add dynamic number of convolutional layers
        for _ in range(num_conv_layers - 1):
            model.add(Conv1D(filters=filters, kernel_size=3, strides=1, padding='same'))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.01))
            model.add(MaxPooling1D(pool_size=3, strides=2, padding='valid'))

        # Add GRU layer
        model.add(Permute((2, 1)))
        model.add(GRU(units=units, dropout=0.5, return_sequences=True))
        model.add(Permute((2, 1)))

        model.add(GlobalAveragePooling1D())

        # Add Dense layer
        model.add(Dense(dense_units, activation='relu'))

        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=2e-4),
                      loss='sparse_categorical_crossentropy',  # Ensure target labels are integer indices
                      metrics=['accuracy'])

        return model

def perform_nas(input_shape, num_classes, X_train, y_train, X_val, y_val, grid):
    # Define the hypermodel
    hypermodel = DynamicAudioCNNHyperModel(input_shape=input_shape, num_classes=num_classes)

    # Create a tuner for random search
    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=3,  # Number of different models to try
        executions_per_trial=1,
        directory='tuner_results',
        project_name=f'power_cnn_{grid}'
    )
    # from tensorflow.keras.utils import to_categorical
    #
    # y_train = to_categorical(y_train, num_classes=num_classes)
    # y_val = to_categorical(y_val, num_classes=num_classes)

    # Search for the best hyperparameters
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Save the model summary to a text file
    summary_file = f'models/Power/{grid}_best_model_summary.txt'
    with open(summary_file, 'w') as f:
        best_model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Train the best model
    history = best_model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val))

    return history, best_model

if __name__ == '__main__':
    for grid in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        # Load the dataset
        training_dataset = np.load(f'saves/power/train_{grid}_vs_all.npz')
        X_train = training_dataset['X_train']
        X_val = training_dataset['X_val']
        y_train = training_dataset['y_train']
        y_val = training_dataset['y_val']

        # Add a new dimension to represent the single channel
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)

        input_shape = X_train[0].shape
        num_classes = np.max(y_train) + 1

        # Perform NAS
        history, best_model = perform_nas(input_shape, num_classes, X_train, y_train, X_val, y_val, grid)

        # Save the final model
        best_model.save(f'models/Power/{grid}vALL_classifier_model.h5')

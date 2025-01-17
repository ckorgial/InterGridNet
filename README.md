# InterGridNet: An Electric Network Frequency Approach for Audio Source Location Classification Using Convolutional Neural Networks 

## **Table of Contents**
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Model Testing](#model-testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## **Introduction**

**InterGridNet** introduces a CNN-based solution for audio source classification using ENF characteristics. The project is designed to handle raw audio recordings, process ENF signals, and classify sources across different grids.

---

## **Project Structure**

The repository is organized as follows:

InterGridNet/ ├── detect_freq_ENF.py # Detect ENF frequencies and prepare training data ├── preprocessing.py # Preprocess audio recordings and normalize data ├── separete_wav.py # Split large audio files into smaller segments ├── split_files.py # Further segmentation of audio files by duration ├── test_detectFreq.py # Evaluate the model's ENF detection accuracy ├── test_model.py # Test the trained model with a confusion matrix ├── train_model_tuner.py # Train the CNN model using hyperparameter tuning ├── models/ # Saved models and summaries ├── databases/ # Input datasets and processed data └── saves/ # Training results and intermediate outputs

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/InterGridNet.git
   cd InterGridNet






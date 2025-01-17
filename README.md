# InterGridNet: An Electric Network Frequency Approach for Audio Source Location Classification Using Convolutional Neural Networks 

## **Table of Contents**
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## **Introduction**

**InterGridNet** introduces a CNN-based solution for audio source classification using ENF characteristics. The project is designed to handle raw audio recordings, process ENF signals, and classify sources across different grids.


## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/InterGridNet.git
   cd InterGridNet

## **Data Preparation**

1. Place raw audio recordings in the databases/database_raw directory.
2. Split large audio files into smaller segments:
```bash
   python separete_wav.py
   python split_files.py
4. Normalize and prepare the dataset:
   ```bash
  python preprocessing.py

## **Model Training**





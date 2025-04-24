# InterGridNet: An Electric Network Frequency Approach for Audio Source Location Classification Using Convolutional Neural Networks 

Implementation of InterGridNet, a RawNet-based framework for audio source location classification using Electric Network Frequency (ENF) features, as proposed by Christos Korgialas et al. in [*InterGridNet: An Electric Network Frequency Approach for Audio Source Location Classification Using Convolutional Neural Networks*](https://www.thinkmind.org/library/SIGNAL/SIGNAL_2025/signal_2025_2_20_60016.html), presented at SIGNAL 2025.

## **Table of Contents**
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Testing](#model-testing)

## **Introduction**

**InterGridNet** introduces a CNN-based solution for audio source classification using ENF characteristics. The project is designed to handle raw audio recordings, process ENF signals, and classify sources across different grids.


## **Installation**

1. Clone the repository:
   ```shell
   git clone https://github.com/yourusername/InterGridNet.git
   cd InterGridNet

## **Data Preparation**

1. Place raw audio recordings in the databases/database_raw directory.
2. Split large audio files into smaller segments
4. Normalize and prepare the dataset


## **Model Training**

```shell
python train_model_tuner.py
```

## **Model Testing**

```shell
python test_detectFreq.py
```

# Authors
Feel free to send us a message for any issue.

***Christos Korgialas (ckorgial@csd.auth.gr)***


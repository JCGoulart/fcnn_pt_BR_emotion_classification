# Fully Connected Neural Network for Emotion Classification (Portuguese)

## Overview
This project implements a fully connected neural network (FCNN) for emotion classification in Portuguese text, using the [Portuguese Tweets](https://huggingface.co/datasets/fpaulino/portuguese-tweets) dataset from Hugging Face. The model classifies text into three sentiment categories: sadness, joy, and neutral.

## 📑 Table of Contents
- [Fully Connected Neural Network for Emotion Classification (Portuguese)](#fully-connected-neural-network-for-emotion-classification-portuguese)
  - [Overview](#overview)
  - [📑 Table of Contents](#-table-of-contents)
  - [📁 Project Structure](#-project-structure)
  - [📊 Dataset](#-dataset)
  - [🔍 Text Preprocessing](#-text-preprocessing)
  - [🧠 Model Architecture](#-model-architecture)
  - [⚙️ Training Details](#️-training-details)
  - [🚀 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [🛠️ Usage](#️-usage)
    - [📥 Data Preparation](#-data-preparation)
    - [🏋️ Train the Model](#️-train-the-model)
    - [🔍 Test the Model](#-test-the-model)
  - [📈 Performance](#-performance)
  - [📋 Requirements](#-requirements)
  - [👥 Contributing](#-contributing)
  - [📄 License](#-license)
  - [🙏 Acknowledgements](#-acknowledgements)

## 📁 Project Structure
```
fcnn_pt_BR_emotion_classification/
│
├── data/                  # Data directory
│   ├── train_data.parquet # Training data
│   ├── test_data.parquet  # Test data
│   └── validation_data.parquet # Validation data
│
├── model/                 # Model artifacts directory
│   ├── fcnn.keras         # Saved Keras model
│   ├── label_encoder.joblib # Serialized label encoder
│   └── vectorizer.joblib  # Serialized TF-IDF vectorizer
│
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── evaluate_and_save.py # Model evaluation and saving utilities
│   ├── preprocessing.py   # Text preprocessing functions
│   └── training_plot.py   # Training visualization utilities
│
├── explore_data.ipynb     # Notebook for Exploratory Data Analysis (EDA)
├── data_preparation.py    # Script for downloading and preprocessing data
├── model_trainer.py       # Script for training and evaluating the model
├── test_model.py          # Script for testing the model with new inputs (to be created)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 📊 Dataset
This project uses the [Portuguese Tweets](https://huggingface.co/datasets/fpaulino/portuguese-tweets) dataset from Hugging Face, which contains Portuguese text samples labeled with three sentiments:
- 😢 Sadness
- 😄 Joy
- 😐 Neutral

The dataset contains a large number of labeled tweets, divided into training and test sets. A portion of the training set is separated for validation.

## 🔍 Text Preprocessing
The preprocessing pipeline includes:
- Lemmatization: Reducing words to their base form using spaCy (`pt_core_news_md`)
- Lowercasing: Converting all text to lowercase
- Stopword removal: Removing common Portuguese words that carry little meaningful information
- Whitespace normalization: Standardizing spaces between words

The processed text is then vectorized using TF-IDF (Term Frequency-Inverse Document Frequency), which converts the text into numerical features based on word importance.

## 🧠 Model Architecture
The model is a fully connected neural network with the following architecture:

![FCNN Architecture Placeholder](https://via.placeholder.com/1192x400?text=FCNN+Architecture+(3+Classes))

- Input layer: Text vectorized with TF-IDF (sparse matrix converted to dense)
- Hidden layer 1: 1024 neurons with SELU activation, `lecun_normal` initializer, and L2 regularization (0.01)
- Hidden layer 2: 512 neurons with SELU activation, `lecun_normal` initializer, and L2 regularization (0.01)
- Hidden layer 3: 256 neurons with SELU activation, `lecun_normal` initializer, and L2 regularization (0.01)
- Hidden layer 4: 64 neurons with SELU activation
- Output layer: 3 neurons with softmax activation (one for each sentiment)

The model uses the SELU (Scaled Exponential Linear Unit) activation function, which aids in self-normalizing properties and can lead to better convergence during training.

## ⚙️ Training Details
- Loss function: Categorical Crossentropy (`CategoricalCrossentropy`)
- Optimizer: Adam
- Learning rate: Starts at 0.001 with step decay (halved every 10 epochs)
- Class weights: Balanced using `compute_class_weight` to handle class imbalance
- Early stopping: Monitored on validation loss (`val_loss`) with a patience of 3 epochs
- Batch size: 256
- Maximum epochs: 20

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Steps
1. Clone the repository:
```bash
git clone https://github.com/yourusername/fcnn_pt_BR_emotion_classification.git # Replace with the correct URL
cd fcnn_pt_BR_emotion_classification
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Download the spaCy model for Portuguese (automatically done in `preprocessing.py`, but can be executed separately):
```bash
python -m spacy download pt_core_news_md
```

## 🛠️ Usage

### 📥 Data Preparation
Download and preprocess the dataset:
```bash
python data_preparation.py
```
This script will:
- Download the Portuguese Tweets dataset from Hugging Face
- Process the text data (lemmatization, stopword removal, etc.)
- Save the processed data as parquet files in the `data/` directory

### 🏋️ Train the Model
Train the FCNN model with the preprocessed data:
```bash
python model_trainer.py
```
This script will:
- Load the preprocessed data
- Apply TF-IDF vectorization
- Train the model with the specified architecture and hyperparameters
- Display training graphs showing loss curves
- Evaluate the model on the test set
- Save the trained model in `model/fcnn.keras`

### 🔍 Test the Model
Test the trained model with new text inputs (create `test_model.py`):
```bash
python test_model.py
```
This script (to be created) will allow testing the model with Portuguese sentences.

## 📈 Performance
The model's performance metrics (accuracy, precision, recall, F1-score) are evaluated on the test dataset and displayed after training. Typical performance for this architecture shows:

- Overall accuracy: ~65-70%
- Precision and recall vary by emotion class

![FCNN Classification Results](model/fcnn_model.png)

The confusion matrix helps visualize which emotions are most often misclassified and which ones the model predicts with highest confidence.

## 📋 Requirements
- Python 3.8+
- TensorFlow
- scikit-learn
- pandas
- numpy
- spaCy (with `pt_core_news_md` model)
- matplotlib
- joblib
- nltk (for stopwords)
- pyarrow (to read parquet)
- emoji

*(Check `requirements.txt` for specific versions)*

## 👥 Contributing
Contributions to improve the model architecture, preprocessing, or add new features are welcome! Feel free to submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License
[MIT License](LICENSE)

## 🙏 Acknowledgements
- [fpaulino](https://huggingface.co/fpaulino) for providing the Portuguese Tweets dataset
- [SpaCy](https://spacy.io/) for natural language processing utilities
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the deep learning framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [NLTK](https://www.nltk.org/) for Portuguese stopwords
- [Hugging Face](https://huggingface.co/) for the datasets platform
- [Netron](https://github.com/lutzroeder/netron) for model visualization
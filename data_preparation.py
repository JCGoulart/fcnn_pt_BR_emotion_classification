# Imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Internal imports
from utils.preprocessing import preprocess_text, nlp

# This script prepares the data for the fully connected neural network
# The dataset used is the Emotion Dataset from Hugging Face
# The dataset is used for emotion classification

# The dataset is available at https://huggingface.co/datasets/fpaulino/portuguese-tweets
# Set train, validation and test splits
# The validation data proved insufficient for the model, so we will use
# 20% of the training data for validation with train_test_split

splits = {'train': 'data/train-00000-of-00001.parquet', 
          'validation': 'data/validation-00000-of-00001.parquet', 
          'test': 'data/test-00000-of-00001.parquet'}

# Load the datasets
train_data = pd.read_parquet("hf://datasets/fpaulino/portuguese-tweets/" + splits["train"])
test_data = pd.read_parquet("hf://datasets/fpaulino/portuguese-tweets/" + splits["test"])

# Remove unused columns
columns_to_remove = ['id', 'tweet_date', 'query_used']

train_data = train_data.drop(columns=columns_to_remove)
test_data = test_data.drop(columns=columns_to_remove)

# Rename columns
rename_dict = {'tweet_text': 'text', 'sentiment': 'label'}

train_data = train_data.rename(columns=rename_dict)
test_data = test_data.rename(columns=rename_dict)

# Rename the labels in the dataframes
labels = {0: 'sadness',
          1: 'joy',
          2: 'neutral'}

# Map the labels to the corresponding values
train_data['label'] = train_data['label'].map(labels)
test_data['label'] = test_data['label'].map(labels)

# Preprocess the text data
train_data['text'] = train_data['text'].apply(lambda x: preprocess_text(x, nlp))
test_data['text'] = test_data['text'].apply(lambda x: preprocess_text(x, nlp))

# Split the training data into training and validation sets
train_data, validation_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Reset the index of the dataframes
train_data = train_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)

# Save the dataframes to parquet files
train_data.to_parquet("data/train_data.parquet")
test_data.to_parquet("data/test_data.parquet")
validation_data.to_parquet("data/validation_data.parquet")
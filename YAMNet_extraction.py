import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# Path to the UrbanSound8K dataset
dataset_path = '/home/nathanael-seay/Downloads/Urban Sound/UrbanSound8K'
metadata_path = os.path.join(dataset_path, 'metadata', 'UrbanSound8K.csv')
audio_folder_path = os.path.join(dataset_path, 'audio')

# Load the metadata file
metadata = pd.read_csv(metadata_path)

# Load YAMNet model from TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

def extract_yamnet_features(file_name):
    try:
        # Load the audio file
        audio_data, sr = librosa.load(file_name, sr=16000)  # YAMNet expects 16kHz audio
        # Ensure audio is 1D
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        # Extract YAMNet embeddings
        scores, embeddings, spectrogram = yamnet_model(audio_data)
        # Average the embeddings
        embeddings = tf.reduce_mean(embeddings, axis=0).numpy()
        return embeddings
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

# Extract features and store in a list
features = []
labels = []

for index, row in metadata.iterrows():
    file_name = os.path.join(audio_folder_path, f'fold{row["fold"]}', row["slice_file_name"])
    class_label = row["class"]
    data = extract_yamnet_features(file_name)
    if data is not None:
        features.append(data)
        labels.append(class_label)

# Convert to a DataFrame
features_df = pd.DataFrame(features)
labels_df = pd.Series(labels, name='label')

# Combine features and labels
final_df = pd.concat([features_df, labels_df], axis=1)

# Save the DataFrame to a CSV file
final_df.to_csv('urbansound8k_yamnet_features.csv', index=False)

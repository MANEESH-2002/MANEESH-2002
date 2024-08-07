import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None

# Example for processing multiple audio files
def process_dataset(file_list):
    features = []
    for file in file_list:
        feature = extract_features(file)
        if feature is not None:
            features.append(feature)
    return np.array(features)

# List of audio files (replace with your own dataset)
audio_files = ['scream1.wav', 'scream2.wav', 'non_scream1.wav', 'non_scream2.wav']
X = process_dataset(audio_files)

# Labels (1 for scream, 0 for non-scream)
y = np.array([1, 1, 0, 0])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

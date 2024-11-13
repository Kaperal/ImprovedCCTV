import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and extract features from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Take the mean of MFCC coefficients
    return mfcc_mean

# Example dataset (replace these with your file paths)
scream_files = ['scream1.wav', 'scream2.wav']
gunshot_files = ['gunshot1.wav', 'gunshot2.wav']
background_files = ['background1.wav', 'background2.wav']

# Prepare dataset
X = []
y = []

for file in scream_files:
    features = extract_features(file)
    X.append(features)
    y.append('scream')

for file in gunshot_files:
    features = extract_features(file)
    X.append(features)
    y.append('gunshot')

for file in background_files:
    features = extract_features(file)
    X.append(features)
    y.append('background')

# Convert to numpy array
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Real-time prediction
import sounddevice as sd

def real_time_detection(duration=2, sr=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("Recording finished.")
    audio = audio.flatten()
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    prediction = model.predict(mfcc_mean)
    print(f"Detected Sound: {prediction[0]}")

# Uncomment the line below to test real-time detection
# real_time_detection()
# train_spss.py

import os
import numpy as np
import librosa
import pickle
from hmmlearn import hmm

DATASET_PATH = 'datasemua'
MODEL_PATH = 'models_spss'
os.makedirs(MODEL_PATH, exist_ok=True)

word_list = ["aku", "kamu", "makan", "selamat", "siang", "malam", "pagi", "sore", "kita", "mereka"]
speaker_id = "all"  # bisa disesuaikan

def extract_mfcc(file):
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
    return mfcc

def train_hmm_models():
    for word in word_list:
        features = []
        folder = os.path.join(DATASET_PATH, speaker_id)
        for fname in os.listdir(folder):
            if fname.lower().startswith(word):
                filepath = os.path.join(folder, fname)
                mfcc = extract_mfcc(filepath)
                features.append(mfcc)
        
        if features:
            X = np.concatenate(features)
            lengths = [len(f) for f in features]
            model = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=100)
            model.fit(X, lengths)
            with open(f"{MODEL_PATH}/{word}.pkl", "wb") as f:
                pickle.dump(model, f)
            print(f"✅ HMM untuk '{word}' dilatih.")
        else:
            print(f"[!] Tidak ada data untuk '{word}'.")

if __name__ == "__main__":
    train_hmm_models()

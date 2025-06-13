import os
import numpy as np
import librosa
import pickle
from hmmlearn import hmm
from pydub import AudioSegment

DATASET_PATH = 'datatrain'
MODEL_PATH = 'models_spss'
os.makedirs(MODEL_PATH, exist_ok=True)

word_list = ["aku", "kamu", "makan", "selamat", "siang", "malam", "pagi", "sore", "kita", "mereka"]
speaker_id = "aul"  # bisa diganti

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

def synthesize_text(text):
    words = text.lower().split()
    output = AudioSegment.silent(duration=0)

    for word in words:
        try:
            with open(f"{MODEL_PATH}/{word}.pkl", "rb") as f:
                model = pickle.load(f)
            generated_mfcc, _ = model.sample(50)  # sampel MFCC
            # Ini hanya dummy untuk demo, kita pakai 1 sample asli terdekat untuk representasi audio
            folder = os.path.join(DATASET_PATH, speaker_id)
            for fname in os.listdir(folder):
                if fname.lower().startswith(word):
                    output += AudioSegment.from_wav(os.path.join(folder, fname))
                    break
        except FileNotFoundError:
            print(f"[!] Model untuk '{word}' tidak ditemukan.")
    
    output.export("output_spss.wav", format="wav")
    print("✅ Output SPSS disimpan sebagai 'output_spss.wav'.")

# Jalankan
if __name__ == "__main__":
    train_hmm_models()
    text = input("Masukkan teks untuk SPSS TTS:\n> ")
    synthesize_text(text)

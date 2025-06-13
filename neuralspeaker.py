import torch
import numpy as np
from pydub import AudioSegment
#from model import SimpleNeuralTTS  # atau model kamu sendiri
from sklearn.preprocessing import LabelEncoder
import librosa
import os

WORDS = ["aku", "kamu", "makan", "selamat", "siang", "malam", "pagi", "sore", "kita", "mereka"]

# Load semua model
models = {
    "afin": torch.load("neural_model_afin.pth"),
    "aul": torch.load("neural_model_aul.pth"),
    "hani": torch.load("neural_model_hani.pth")
}

# Encoder untuk kata
word_encoder = LabelEncoder().fit(WORDS)

# Path ke data asli per speaker (untuk mencari audio terdekat)
DATASET_PATH = "datatrain"

# Fungsi ekstraksi MFCC
def extract_mfcc(path):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Fungsi sintesis
def synthesize(text, speaker_id):
    model = models.get(speaker_id.lower())
    if not model:
        print(f"[!] Speaker '{speaker_id}' tidak ditemukan.")
        return

    words = text.lower().split()
    output_audio = AudioSegment.silent(duration=0)

    for word in words:
        if word not in WORDS:
            print(f"[!] Kata '{word}' tidak dikenal.")
            continue
        word_id = torch.LongTensor([word_encoder.transform([word])[0]])
        with torch.no_grad():
            predicted_mfcc = model(word_id).numpy().squeeze()

        # Cari audio sample terdekat
        folder = os.path.join(DATASET_PATH, speaker_id)
        best_sample = None
        min_dist = float("inf")
        for fname in os.listdir(folder):
            if fname.lower().startswith(word):
                path = os.path.join(folder, fname)
                mfcc = extract_mfcc(path)
                dist = np.linalg.norm(mfcc - predicted_mfcc)
                if dist < min_dist:
                    min_dist = dist
                    best_sample = path

        if best_sample:
            audio = AudioSegment.from_wav(best_sample)
            output_audio += audio + AudioSegment.silent(duration=100)
        else:
            print(f"[!] Tidak ada audio untuk kata '{word}' pada speaker '{speaker_id}'.")

    output_audio.export(f"output_neural_tts_{speaker_id}.wav", format="wav")
    print(f"✅ Output disimpan sebagai 'output_neural_tts_{speaker_id}.wav'")

# Main Program
if __name__ == "__main__":
    text = input("Masukkan kalimat: ")
    speaker = input("Pilih speaker (afin/aul/hani): ")
    synthesize(text, speaker)

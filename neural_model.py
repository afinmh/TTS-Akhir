import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "datatrain"
WORDS = ["aku", "kamu", "makan", "selamat", "siang", "malam", "pagi", "sore", "kita", "mereka"]

# Model sederhana: Text ID → MFCC
class SimpleTTSModel(nn.Module):
    def __init__(self, vocab_size, output_dim):
        super(SimpleTTSModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        return self.fc(x)

# Ekstraksi MFCC rata-rata
def extract_mfcc(filepath):
    y, sr = librosa.load(filepath, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Fungsi training
def train_model_for_speaker(speaker_id):
    print(f"🔧 Melatih model untuk speaker '{speaker_id}'...")
    folder = os.path.join(DATASET_PATH, speaker_id)

    if not os.path.exists(folder):
        print(f"[!] Folder speaker '{speaker_id}' tidak ditemukan.")
        return

    le = LabelEncoder()
    le.fit(WORDS)

    X, y = [], []
    for fname in os.listdir(folder):
        for word in WORDS:
            if fname.lower().startswith(word):
                label = le.transform([word])[0]
                mfcc = extract_mfcc(os.path.join(folder, fname))
                X.append(label)
                y.append(mfcc)

    if not X:
        print(f"[!] Tidak ada data latih ditemukan untuk speaker '{speaker_id}'.")
        return

    X = torch.LongTensor(X)
    y = torch.Tensor(y)

    model = SimpleTTSModel(len(WORDS), y.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    model_path = f"neural_model_{speaker_id}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model Neural TTS untuk speaker '{speaker_id}' disimpan sebagai '{model_path}'.")

# 🏁 Run
if __name__ == "__main__":
    speaker = input("Masukkan ID speaker (afin/aul/hani): ").strip().lower()
    train_model_for_speaker(speaker)

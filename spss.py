# synthesize_spss.py

import os
import pickle
from pydub import AudioSegment

DATASET_PATH = 'datasemua'
MODEL_PATH = 'models_spss'
word_list = ["aku", "kamu", "makan", "selamat", "siang", "malam", "pagi", "sore", "kita", "mereka"]
speaker_id = "all"  # bisa disesuaikan

def synthesize_text(text):
    words = text.lower().split()
    output = AudioSegment.silent(duration=0)

    for word in words:
        try:
            with open(f"{MODEL_PATH}/{word}.pkl", "rb") as f:
                model = pickle.load(f)
            
            # Dummy: Ambil audio nyata pertama yang sesuai
            folder = os.path.join(DATASET_PATH, speaker_id)
            for fname in os.listdir(folder):
                if fname.lower().startswith(word):
                    output += AudioSegment.from_wav(os.path.join(folder, fname))
                    break
        except FileNotFoundError:
            print(f"[!] Model untuk '{word}' tidak ditemukan.")
    
    output.export("output_spss.wav", format="wav")
    print("✅ Output SPSS disimpan sebagai 'output_spss.wav'.")

if __name__ == "__main__":
    text = input("Masukkan teks untuk SPSS TTS:\n> ")
    synthesize_text(text)

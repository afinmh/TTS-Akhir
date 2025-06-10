import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import os

# Konfigurasi
DURATION = 3  # durasi rekaman dalam detik
FS = 16000    # sample rate
OUTPUT_PATH = "output/audio_converted.wav"

def record_audio(duration, fs):
    print(f"[INFO] Merekam audio selama {duration} detik...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio

def save_as_wav(filename, audio, fs):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    wavfile.write(filename, fs, audio)
    print(f"[INFO] Disimpan sebagai WAV: {filename}")

if __name__ == "__main__":
    audio = record_audio(DURATION, FS)
    save_as_wav(OUTPUT_PATH, audio, FS)
    print("[SELESAI]")

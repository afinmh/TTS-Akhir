import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import os
from datetime import datetime

# Konfigurasi rekaman
DURATION = 2  # detik
FS = 16000    # sampling rate
KATA_UNIK = ["aku", "kamu", "makan", "selamat", "siang", "malam", "pagi", "sore", "kita", "mereka"]
NAMA_ANGGOTA = ["afin", "aul", "hani"]

def buat_folder(jenis_data, nama):
    path = os.path.join(jenis_data, nama)
    os.makedirs(path, exist_ok=True)
    return path

def rekam_suara(filename):
    print(f"[INFO] Merekam selama {DURATION} detik...")
    try:
        audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()  # tunggu hingga rekaman selesai
        wavfile.write(filename, FS, audio)
        print(f"[SAVED] {filename}\n")
    except Exception as e:
        print(f"[ERROR] Gagal merekam: {e}")

def main():
    print("=== Perekaman Dataset TTS ===")
    jenis_data = input("Masukkan jenis data (datatrain/datatest): ").strip().lower()
    if jenis_data not in ["datatrain", "datatest"]:
        print("Jenis data tidak valid.")
        return

    nama = input("Masukkan nama (afin/aul/hani): ").strip().lower()
    if nama not in NAMA_ANGGOTA:
        print("Nama tidak terdaftar.")
        return

    path = buat_folder(jenis_data, nama)

    print("\nDaftar Kata:")
    for i, kata in enumerate(KATA_UNIK):
        print(f"{i+1}. {kata}")
    print("")

    while True:
        try:
            kata_index = int(input("Rekam kata ke (1-10, 0 untuk keluar): "))
            if kata_index == 0:
                break
            if not 1 <= kata_index <= 10:
                print("Index tidak valid.")
                continue

            kata = KATA_UNIK[kata_index - 1]
            jumlah = int(input(f"Rekam berapa sample untuk kata '{kata}': "))

            for i in range(jumlah):
                input(f"Tekan ENTER untuk mulai rekaman ke-{i+1} untuk kata '{kata}'...")
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = os.path.join(path, f"{kata}_{i+1}_{timestamp}.wav")
                rekam_suara(filename)

        except Exception as e:
            print(f"Terjadi kesalahan: {e}")

    print("Perekaman selesai.")

if __name__ == "__main__":
    main()

import os
from pydub import AudioSegment

# Path ke dataset training
DATASET_PATH = 'datatrain'
# Kata-kata unik (lowercase agar fleksibel input pengguna)
word_list = ["aku", "kamu", "makan", "selamat", "siang", "malam", "pagi", "sore", "kita", "mereka"]

# Pilih satu contoh terbaik dari masing-masing kata
def select_best_sample(speaker_folder, word):
    samples = [f for f in os.listdir(speaker_folder) if f.lower().startswith(word) and f.endswith('.wav')]
    samples = sorted(samples)
    if samples:
        filepath = os.path.join(speaker_folder, samples[0])
        return AudioSegment.from_wav(filepath)
    else:
        return None

# Fungsi utama untuk menggabungkan audio berdasarkan input teks
def synthesize_text(text, speaker_id="afin"):
    words = text.lower().split()
    output_audio = AudioSegment.silent(duration=0)

    speaker_folder = os.path.join(DATASET_PATH, speaker_id)
    if not os.path.isdir(speaker_folder):
        print(f"[!] Folder speaker '{speaker_folder}' tidak ditemukan.")
        return None

    print(f"[i] Menyintesis teks: {' '.join(words)} (Speaker: {speaker_id})")

    for word in words:
        if word in word_list:
            audio = select_best_sample(speaker_folder, word)
            if audio:
                output_audio += audio + AudioSegment.silent(duration=100)
            else:
                print(f"[!] Sample untuk kata '{word}' tidak ditemukan.")
        else:
            print(f"[!] Kata '{word}' tidak ada di kamus.")

    return output_audio

# Main program
if __name__ == "__main__":
    # Input dari pengguna
    input_text = input("Masukkan teks (gunakan kata-kata: Aku, Kamu, Makan, Selamat, Siang, Malam, Pagi, Sore, Kita, Mereka):\n> ")
    speaker_id = input("Masukkan speaker ID (afin/aul/hani):\n> ").lower()

    if speaker_id not in ['afin', 'aul', 'hani']:
        print("[!] Speaker ID tidak valid.")
    else:
        output = synthesize_text(input_text, speaker_id=speaker_id)
        if output:
            filename = f"output_tts_{speaker_id}.wav"
            output.export(filename, format="wav")
            print(f"✅ Output disimpan sebagai '{filename}'")

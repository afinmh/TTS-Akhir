import os
import random
from pydub import AudioSegment
from pydub.playback import play

# --- KONFIGURASI ---
# Sesuaikan path ini dengan nama folder tempat Anda menyimpan file .wav
DATABASE_PATH = "database_suara" 

def load_audio_database(path):
    """
    Memindai direktori dan membuat database kata dan path filenya.
    Database berbentuk dictionary: {'kata': ['path/ke/file1.wav', 'path/ke/file2.wav']}
    """
    audio_db = {}
    print(f"🔄 Memuat database suara dari folder: '{path}'...")
    
    if not os.path.isdir(path):
        print(f"❌ ERROR: Folder '{path}' tidak ditemukan. Pastikan folder sudah dibuat dan berisi file .wav.")
        return None

    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            # Ekstrak kata dari nama file (bagian sebelum underscore pertama)
            word = filename.split('_')[0].lower()
            
            # Tambahkan path file ke dalam list untuk kata tersebut
            full_path = os.path.join(path, filename)
            if word not in audio_db:
                audio_db[word] = []
            audio_db[word].append(full_path)
            
    if not audio_db:
        print("❌ WARNING: Tidak ada file .wav yang ditemukan di dalam database.")
    else:
        print(f"✅ Database berhasil dimuat. {len(audio_db)} kata unik ditemukan.")

    return audio_db

def synthesize_speech(text, db):
    """
    Menerima teks dan database, lalu menggabungkan audio kata per kata.
    """
    # Ubah teks input menjadi huruf kecil dan pisahkan menjadi kata-kata
    words = text.lower().split()
    
    # Siapkan segmen audio kosong sebagai dasar
    final_audio = AudioSegment.empty()
    
    print("\n🔊 Memulai sintesis suara...")
    for word in words:
        if word in db:
            # Jika kata ada di database, pilih salah satu sampel secara acak
            available_samples = db[word]
            selected_file = random.choice(available_samples)
            print(f"   - Menemukan kata '{word}'. Memilih sampel: {os.path.basename(selected_file)}")
            
            # Muat file audio yang dipilih
            audio_segment = AudioSegment.from_wav(selected_file)
            
            # Gabungkan (concatenate) ke audio final
            final_audio += audio_segment
        else:
            # Jika kata tidak ditemukan, beri peringatan
            print(f"   - ⚠️ Peringatan: Kata '{word}' tidak ditemukan dalam database. Kata ini akan dilewati.")
            
    return final_audio

# --- Program Utama ---
if __name__ == "__main__":
    # 1. Muat database audio saat program dimulai
    audio_database = load_audio_database(DATABASE_PATH)
    
    if audio_database:
        print("\n--- Program Text-to-Speech Concatenative ---")
        print("Kata yang tersedia:", ", ".join(sorted(audio_database.keys())))
        print("Ketik 'keluar' untuk mengakhiri program.")
        
        while True:
            # 2. Minta input dari pengguna
            try:
                input_text = input("\nMasukkan kalimat > ")
            except KeyboardInterrupt:
                print("\nProgram dihentikan.")
                break

            if input_text.lower() == 'keluar':
                print("👋 Sampai jumpa!")
                break
            
            if not input_text.strip():
                continue

            # 3. Lakukan sintesis suara
            synthesized_audio = synthesize_speech(input_text, audio_database)
            
            # 4. Putar hasilnya jika ada audio yang berhasil dibuat
            if len(synthesized_audio) > 0:
                print("🎶 Memutar hasil suara...")
                play(synthesized_audio)
            else:
                print("🔇 Tidak ada suara yang dihasilkan karena semua kata tidak ditemukan.")
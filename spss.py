import os
import numpy as np
import librosa
import sounddevice as sd
import joblib # Untuk menyimpan dan memuat model
from hmmlearn import hmm

# --- KONFIGURASI ---
DATABASE_PATH = "database_suara"
MODELS_PATH = "models"
N_MFCC = 13       # Jumlah koefisien MFCC yang akan diekstrak
N_COMPONENTS = 5  # Jumlah 'states' dalam setiap HMM. Anda bisa bereksperimen dengan angka ini.
SAMPLING_RATE = 22050 # Standard sampling rate untuk audio

def extract_features(file_path):
    """Mengekstrak fitur MFCC dari sebuah file audio."""
    try:
        # Muat file audio dengan librosa, pastikan sampling rate seragam
        y, sr = librosa.load(file_path, sr=SAMPLING_RATE)
        # Ekstrak MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        # Transpose agar baris adalah waktu dan kolom adalah fitur
        return mfccs.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def train_and_save_models():
    """Melatih model HMM untuk setiap kata dalam database dan menyimpannya."""
    # 1. Mengelompokkan file berdasarkan kata
    word_files = {}
    print("📂 Mengelompokkan file audio berdasarkan kata...")
    for filename in os.listdir(DATABASE_PATH):
        if filename.endswith(".wav"):
            word = filename.split('_')[0].lower()
            if word not in word_files:
                word_files[word] = []
            word_files[word].append(os.path.join(DATABASE_PATH, filename))

    if not word_files:
        print("❌ Tidak ada file .wav ditemukan. Proses training dibatalkan.")
        return False

    # Membuat folder models jika belum ada
    os.makedirs(MODELS_PATH, exist_ok=True)

    # 2. Melatih satu HMM untuk setiap kata
    print("\n🏋️  Memulai pelatihan model HMM...")
    for word, files in word_files.items():
        print(f"  -> Melatih model untuk kata: '{word}'")
        
        # Kumpulkan fitur dari semua sampel untuk kata ini
        features = []
        lengths = []
        for file_path in files:
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                lengths.append(len(mfccs))

        if not features:
            print(f"     ⚠️ Tidak ada fitur yang bisa diekstrak untuk kata '{word}'. Dilewati.")
            continue
        
        # Gabungkan semua fitur menjadi satu array besar
        all_features_for_word = np.concatenate(features)

        # 3. Inisialisasi dan latih model HMM
        model = hmm.GaussianHMM(n_components=N_COMPONENTS, covariance_type="diag", n_iter=100)
        model.fit(all_features_for_word, lengths=lengths)

        # 4. Simpan model yang sudah dilatih
        model_path = os.path.join(MODELS_PATH, f"{word}.pkl")
        joblib.dump(model, model_path)
        print(f"     ✅ Model untuk '{word}' disimpan di {model_path}")
        
    print("\n✨ Pelatihan semua model selesai.")
    return True

def features_to_audio(mfccs, sr=SAMPLING_RATE):
    """Mengubah fitur MFCC kembali menjadi audio menggunakan algoritma Griffin-Lim."""
    # Transpose kembali agar sesuai dengan format librosa
    mfccs_T = mfccs.T
    # Gunakan Griffin-Lim untuk merekonstruksi audio dari spektogram (inversi MFCC)
    # Ini adalah proses aproksimasi dan mungkin menghasilkan suara robotik
    y_reconstructed = librosa.feature.inverse.mfcc_to_audio(mfccs_T)
    return y_reconstructed

def synthesize_speech(text):
    """Mensintesis ucapan dari teks menggunakan model HMM yang telah dilatih."""
    words = text.lower().split()
    final_audio_features = []
    
    print("\n🔊 Memulai sintesis suara...")
    for word in words:
        model_path = os.path.join(MODELS_PATH, f"{word}.pkl")
        if os.path.exists(model_path):
            # Muat model HMM untuk kata tersebut
            model = joblib.load(model_path)
            
            # Tentukan panjang rata-rata sequence untuk kata ini dari model
            # (Simplifikasi: kita ambil dari panjang rata-rata saat training)
            # Untuk hasil lebih baik, ini seharusnya dimodelkan juga.
            # Kita gunakan panjang rata-rata dari matriks transisi.
            n_samples = 15 * N_COMPONENTS # Estimasi panjang sequence
            
            # Hasilkan (generate) urutan fitur dari model
            features, _ = model.sample(n_samples)
            print(f"   - Menghasilkan fitur untuk kata '{word}'")
            final_audio_features.append(features)
        else:
            print(f"   - ⚠️ Peringatan: Model untuk kata '{word}' tidak ditemukan. Dilewati.")

    if not final_audio_features:
        return None

    # Gabungkan semua fitur yang dihasilkan
    full_feature_sequence = np.concatenate(final_audio_features)
    
    # Ubah sekuens fitur kembali menjadi audio
    print("   - Mengonversi fitur menjadi audio (Griffin-Lim)...")
    synthesized_audio = features_to_audio(full_feature_sequence)
    return synthesized_audio

# --- Program Utama ---
if __name__ == "__main__":
    print("--- Program SPSS berbasis HMM ---")
    
    # Tanya pengguna apakah mau melakukan training atau tidak
    choice = input("Apakah Anda ingin memulai training model baru? (y/n): ").lower()
    
    if choice == 'y':
        training_success = train_and_save_models()
        if not training_success:
            exit() # Keluar jika training gagal
            
    # Periksa apakah ada model yang bisa digunakan
    if not os.path.exists(MODELS_PATH) or not os.listdir(MODELS_PATH):
        print("\n❌ Tidak ada model yang ditemukan. Silakan jalankan training terlebih dahulu.")
        exit()

    print("\n✅ Model HMM siap digunakan.")
    
    while True:
        try:
            input_text = input("\nMasukkan kalimat untuk disintesis (atau 'keluar'): ")
            if input_text.lower() == 'keluar':
                print("👋 Sampai jumpa!")
                break
            
            if not input_text.strip():
                continue

            # Lakukan sintesis
            audio_output = synthesize_speech(input_text)

            if audio_output is not None:
                print("🎶 Memutar hasil suara...")
                # Putar audio menggunakan sounddevice
                sd.play(audio_output, SAMPLING_RATE)
                sd.wait() # Tunggu sampai audio selesai diputar
            else:
                print("🔇 Tidak ada suara yang dihasilkan.")

        except KeyboardInterrupt:
            print("\nProgram dihentikan.")
            break
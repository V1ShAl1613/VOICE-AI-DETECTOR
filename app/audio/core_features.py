import librosa
import numpy as np
import scipy.stats
import os
import tempfile

def extract_features(file_path: str):
    """
    Extracts comprehensive acoustic features from an audio file using librosa.
    Returns a 1D numpy array of features.
    
    Feature vector breakdown (~83 features):
      - MFCC Mean (13) + Std (13) = 26
      - Delta MFCC Mean (13) + Std (13) = 26
      - Spectral Centroid Mean + Std = 2
      - Spectral Rolloff Mean + Std = 2
      - Spectral Flatness Mean = 1
      - Spectral Bandwidth Mean + Std = 2
      - Spectral Contrast Mean (7) = 7
      - Chroma STFT Mean (12) = 12
      - Pitch Mean + Std = 2
      - ZCR Mean + Var = 2
      - RMSE Mean + Var = 2
      - Silence Ratio = 1
      - Spectral Smoothness = 1
      - Tonnetz Mean (6) = 6
      Total = ~92 features
    """
    try:
        # Load audio directly with librosa (supports MP3 via audioread/soundfile)
        try:
            y, sr = librosa.load(file_path, sr=22050, mono=True)
        except Exception as e:
            raise ValueError(f"Cannot decode audio file: {str(e)}")
        
        # Force data into memory
        y = np.array(y, copy=True)
        
        # Validate audio
        if len(y) == 0:
            raise ValueError("Audio file is empty.")
        
        duration_sec = len(y) / sr
        if duration_sec < 0.5:
            raise ValueError(f"Audio too short ({duration_sec:.2f}s). Minimum 0.5 seconds required.")
        
        # ===================== CORE FEATURES =====================
        
        # 1. MFCCs (Mean + Std) - 13 coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)       # 13
        mfcc_std = np.std(mfcc, axis=1)          # 13
        
        # 2. Delta MFCCs (1st order) - temporal dynamics
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)   # 13
        delta_mfcc_std = np.std(delta_mfcc, axis=1)      # 13
        
        # 3. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)         # 1
        centroid_std = np.std(centroid)            # 1
        
        # 4. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)           # 1
        rolloff_std = np.std(rolloff)             # 1
        
        # 5. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)         # 1
        
        # 6. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(bandwidth)       # 1
        bandwidth_std = np.std(bandwidth)         # 1
        
        # 7. Spectral Contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1) # 7
        
        # 8. Chroma STFT (12 bins)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)     # 12
        
        # ===================== PROSODIC FEATURES =====================
        
        # 9. Pitch (F0) Analysis using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches_indices = magnitudes > np.median(magnitudes)
        pitch_values = pitches[pitches_indices]
        pitch_values = pitch_values[pitch_values > 0]
        
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values)    # 1
            pitch_std = np.std(pitch_values)      # 1
        else:
            pitch_mean = 0
            pitch_std = 0
            
        # 10. Zero Crossing Rate (Jitter proxy)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)                   # 1
        zcr_var = np.var(zcr)                     # 1
        
        # 11. RMSE (Energy/Amplitude - Shimmer proxy)
        rmse = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rmse)                 # 1
        rmse_var = np.var(rmse)                   # 1
        
        # 12. Silence Ratio
        rmse_db = librosa.amplitude_to_db(rmse, ref=np.max)
        silence_threshold_db = -40
        silence_frames = np.sum(rmse_db < silence_threshold_db)
        total_frames = len(rmse_db[0])
        silence_ratio = silence_frames / total_frames if total_frames > 0 else 0  # 1
        
        # 13. Spectral Smoothness
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        spectral_smoothness = np.mean(np.diff(onset_env))  # 1
        
        # ===================== ADVANCED FEATURES =====================
        
        # 14. Tonnetz (tonal centroid features) - 6 dimensions
        try:
            harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)  # 6
        except Exception:
            tonnetz_mean = np.zeros(6)
        
        # Concatenate all features
        features = np.hstack([
            mfcc_mean,           # 13
            mfcc_std,            # 13
            delta_mfcc_mean,     # 13
            delta_mfcc_std,      # 13
            centroid_mean,       # 1
            centroid_std,        # 1
            rolloff_mean,        # 1
            rolloff_std,         # 1
            flatness_mean,       # 1
            bandwidth_mean,      # 1
            bandwidth_std,       # 1
            contrast_mean,       # 7
            chroma_mean,         # 12
            pitch_mean,          # 1
            pitch_std,           # 1
            zcr_mean,            # 1
            zcr_var,             # 1
            rmse_mean,           # 1
            rmse_var,            # 1
            silence_ratio,       # 1
            spectral_smoothness, # 1
            tonnetz_mean,        # 6
        ])
        # Total: 13+13+13+13+1+1+1+1+1+1+1+7+12+1+1+1+1+1+1+1+1+6 = 92
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {type(e).__name__}: {e}")
        raise ValueError(str(e) if str(e) else f"Audio processing error: {type(e).__name__}")

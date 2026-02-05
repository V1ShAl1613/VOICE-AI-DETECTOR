import librosa
import numpy as np
import scipy.stats
import os
import tempfile

def extract_features(file_path: str):
    """
    Extracts acoustic features from an audio file using librosa.
    Returns a 1D numpy array of features.
    Feature vector size depends on the number of stats computed.
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
        
        # 1. MFCCs (Mean + Std) - 13 coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 2. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)
        
        # 3. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        
        # 4. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)
        
        # 5. Pitch (F0) Analysis using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches_indices = magnitudes > np.median(magnitudes)
        pitch_values = pitches[pitches_indices]
        pitch_values = pitch_values[pitch_values > 0]
        
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
        else:
            pitch_mean = 0
            pitch_std = 0
            
        # 6. Jitter & Shimmer Approximations
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)
        
        # RMSE (Energy/Amplitude)
        rmse = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rmse)
        rmse_var = np.var(rmse)
        
        # 7. Silence Ratio
        rmse_db = librosa.amplitude_to_db(rmse, ref=np.max)
        silence_threshold_db = -40
        silence_frames = np.sum(rmse_db < silence_threshold_db)
        total_frames = len(rmse_db[0])
        silence_ratio = silence_frames / total_frames if total_frames > 0 else 0
        
        # 8. Spectral Smoothness
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        spectral_smoothness = np.mean(np.diff(onset_env))
        
        # Concatenate all features
        features = np.hstack([
            mfcc_mean,       # 13
            mfcc_std,        # 13
            centroid_mean,   # 1
            centroid_std,    # 1
            rolloff_mean,    # 1
            rolloff_std,     # 1
            flatness_mean,   # 1
            pitch_mean,      # 1
            pitch_std,       # 1
            zcr_mean,        # 1
            zcr_var,         # 1
            rmse_mean,       # 1
            rmse_var,        # 1
            silence_ratio,   # 1
            spectral_smoothness # 1
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {type(e).__name__}: {e}")
        raise ValueError(str(e) if str(e) else f"Audio processing error: {type(e).__name__}")

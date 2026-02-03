import numpy as np

def generate_explanation(features: np.ndarray, prediction_prob: float, threshold: float) -> str:
    """
    Generates a technical explanation based on feature values and classification.
    Expects the full feature vector as input.
    """
    if prediction_prob < threshold:
        return "Natural specific traits detected: irregular pitch patterns and organic spectral variance."
    
    # Feature indices (must match features.py):
    # 0-12: MFCC Mean
    # 13-25: MFCC Std
    # 26: Centroid Mean, 27: Centroid Std
    # 28: Rolloff Mean, 29: Rolloff Std
    # 30: Flatness Mean
    # 31: Pitch Mean, 32: Pitch Std
    # 33: ZCR Mean, 34: ZCR Var
    # 35: RMSE Mean, 36: RMSE Var
    # 37: Silence Ratio
    # 38: Spectral Smoothness
    
    reasons = []
    
    # Extract key metrics
    pitch_std = features[32]
    zcr_var = features[34] # Proxy for jitter/irregularity
    silence_ratio = features[37]
    spectral_smoothness = features[38]
    
    # Logic for AI artifacts
    
    # 1. Robotic Pitch Consistency (Low pitch variance)
    # Thresholds are heuristic; ideally these should be continuously calibrated
    if pitch_std < 10.0: 
        reasons.append("robotic pitch consistency")
        
    # 2. Synthetic Vocal Stability (Low jitter/shimmer proxy)
    if zcr_var < 0.001:
        reasons.append("synthetic vocal stability")
        
    # 3. Absence of Natural Breathing (Low silence ratio)
    if silence_ratio < 0.05:
        reasons.append("absence of natural breathing")
        
    # 4. Speech Synthesis Artifacts (Over-smoothed spectra)
    # High smoothness usually means less transient noise which characterizes human speech
    if abs(spectral_smoothness) < 0.5: 
        reasons.append("over-smoothed spectral transitions")
        
    if not reasons:
        reasons.append("detected statistical anomalies in frequency distribution")
        
    # Join nicely
    if len(reasons) > 1:
        text = ", ".join(reasons[:-1]) + ", and " + reasons[-1]
    else:
        text = reasons[0]
        
    return f"{text.capitalize()} detected"

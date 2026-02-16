import numpy as np

def generate_explanation(features: np.ndarray, prediction_prob: float, threshold: float) -> str:
    """
    Generates a technical explanation based on feature values and classification.
    Updated for 92-feature vector layout.
    
    Feature indices (must match core_features.py):
      0-12:  MFCC Mean (13)
      13-25: MFCC Std (13)
      26-38: Delta MFCC Mean (13)
      39-51: Delta MFCC Std (13)
      52: Centroid Mean, 53: Centroid Std
      54: Rolloff Mean, 55: Rolloff Std
      56: Flatness Mean
      57: Bandwidth Mean, 58: Bandwidth Std
      59-65: Spectral Contrast Mean (7)
      66-77: Chroma Mean (12)
      78: Pitch Mean, 79: Pitch Std
      80: ZCR Mean, 81: ZCR Var
      82: RMSE Mean, 83: RMSE Var
      84: Silence Ratio
      85: Spectral Smoothness
      86-91: Tonnetz Mean (6)
    """
    if prediction_prob < threshold:
        # HUMAN classification
        reasons = []
        
        # Check for natural voice indicators
        pitch_std = features[79] if len(features) > 79 else 0
        zcr_var = features[81] if len(features) > 81 else 0
        silence_ratio = features[84] if len(features) > 84 else 0
        
        if pitch_std > 15.0:
            reasons.append("natural pitch variation")
        if zcr_var > 0.001:
            reasons.append("organic vocal irregularities")
        if silence_ratio > 0.05:
            reasons.append("natural breathing patterns")
        
        if not reasons:
            reasons.append("natural human speech characteristics")
        
        if len(reasons) > 1:
            text = ", ".join(reasons[:-1]) + ", and " + reasons[-1]
        else:
            text = reasons[0]
        return f"Human voice indicators detected: {text}."
    
    # AI_GENERATED classification
    reasons = []
    
    # Extract key metrics safely
    pitch_std = features[79] if len(features) > 79 else 0
    zcr_var = features[81] if len(features) > 81 else 0
    silence_ratio = features[84] if len(features) > 84 else 0
    spectral_smoothness = features[85] if len(features) > 85 else 0
    flatness = features[56] if len(features) > 56 else 0
    
    # 1. Robotic Pitch Consistency
    if pitch_std < 10.0: 
        reasons.append("robotic pitch consistency")
        
    # 2. Synthetic Vocal Stability
    if zcr_var < 0.001:
        reasons.append("synthetic vocal stability")
        
    # 3. Absence of Natural Breathing
    if silence_ratio < 0.05:
        reasons.append("absence of natural breathing patterns")
        
    # 4. Over-smoothed Spectra
    if abs(spectral_smoothness) < 0.5: 
        reasons.append("over-smoothed spectral transitions")
    
    # 5. Flat spectral distribution (TTS artifact)
    if flatness > 0.1:
        reasons.append("unnaturally flat spectral distribution")
        
    if not reasons:
        reasons.append("statistical anomalies in frequency distribution consistent with AI synthesis")
        
    if len(reasons) > 1:
        text = ", ".join(reasons[:-1]) + ", and " + reasons[-1]
    else:
        text = reasons[0]
        
    return f"AI-generated voice indicators detected: {text}."

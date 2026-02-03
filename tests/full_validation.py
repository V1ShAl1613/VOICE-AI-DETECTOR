import os
import sys
import glob
import base64
import requests
import joblib
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.audio.features import extract_features
from app.core.config import settings

# Configuration
API_URL = "http://localhost:8002/api/voice-detection"  # Using 8002 as fallback
API_KEY = "secret123"
MODEL_PATH = "app/ml/voice_auth_model.pkl"

class ValidationReporter:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def log_pass(self, test_name):
        print(f"✅ PASS: {test_name}")
        self.passed.append(test_name)

    def log_fail(self, test_name, reason):
        print(f"❌ FAIL: {test_name} - {reason}")
        self.failed.append(f"{test_name}: {reason}")

    def log_warn(self, test_name, reason):
        print(f"⚠️ WARN: {test_name} - {reason}")
        self.warnings.append(f"{test_name}: {reason}")

    def report(self):
        print("\n" + "="*40)
        print("FINAL COMPLIANCE REPORT")
        print("="*40)
        print(f"Passed Tests: {len(self.passed)}")
        print(f"Failed Tests: {len(self.failed)}")
        print(f"Warnings:     {len(self.warnings)}")
        
        if self.failed:
            print("\nFAILURES:")
            for f in self.failed:
                print(f" - {f}")

        if self.warnings:
            print("\nWARNINGS:")
            for w in self.warnings:
                print(f" - {w}")

        print("\nSTATUS: " + ("READY FOR SUBMISSION" if not self.failed else "NOT READY"))
        print("="*40)

reporter = ValidationReporter()

def get_audio_sample(type_):
    # type_ is 'human' or 'ai_generated'
    # Look in ../dataset first
    base_path = os.path.join("dataset", type_)
    files = glob.glob(f"{base_path}/**/*.wav", recursive=True) + glob.glob(f"{base_path}/**/*.mp3", recursive=True)
    if not files:
        # try standard relative path if running from root
        base_path = os.path.join("voice_ai_detector", "dataset", type_)
        files = glob.glob(f"{base_path}/**/*.wav", recursive=True) + glob.glob(f"{base_path}/**/*.mp3", recursive=True)
    
    if files:
        return files[0]
    return None

def test_model_integrity():
    print("\n--- PHASE 2: MODEL INTEGRITY CHECK (OFFLINE) ---")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        # try relative path
        if os.path.exists(f"voice_ai_detector/{MODEL_PATH}"):
            path = f"voice_ai_detector/{MODEL_PATH}"
        else:
            reporter.log_fail("Model File Check", f"Model not found at {MODEL_PATH}")
            return
    else:
        path = MODEL_PATH
        
    try:
        model = joblib.load(path)
        reporter.log_pass("Model Load")
    except Exception as e:
        reporter.log_fail("Model Load", str(e))
        return

    # 2. Prediction Check
    human_file = get_audio_sample("human")
    ai_file = get_audio_sample("ai_generated")
    
    if not human_file or not ai_file:
         reporter.log_warn("Model Prediction Check", "Could not find samples for offline check. Skipping.")
         return

    try:
        # Extract features
        h_feat = extract_features(human_file).reshape(1, -1)
        a_feat = extract_features(ai_file).reshape(1, -1)
        
        # Predict
        if hasattr(model, "predict_proba"):
             h_prob = model.predict_proba(h_feat)[0][1]
             a_prob = model.predict_proba(a_feat)[0][1]
        else:
             h_prob = float(model.predict(h_feat)[0])
             a_prob = float(model.predict(a_feat)[0])
             
        print(f"   Human Prob: {h_prob:.4f}")
        print(f"   AI Prob:    {a_prob:.4f}")
        
        if h_prob < a_prob:
             reporter.log_pass("Model Discrimination (Human < AI)")
        else:
             reporter.log_fail("Model Discrimination", f"Human prob {h_prob} >= AI prob {a_prob}")
             
        threshold = settings.AI_PROBABILITY_THRESHOLD
        if h_prob < threshold:
             reporter.log_pass("Human Classification Correct")
        else:
             reporter.log_fail("Human Classification", f"Classified as AI (Prob {h_prob})")

        if a_prob >= threshold:
             reporter.log_pass("AI Classification Correct")
        else:
             reporter.log_fail("AI Classification", f"Classified as Human (Prob {a_prob})")
             
    except Exception as e:
        reporter.log_fail("Model Inference", str(e))

def test_feature_extraction():
    print("\n--- PHASE 3: FEATURE EXTRACTION VALIDATION ---")
    sample = get_audio_sample("human")
    if not sample:
        reporter.log_warn("Feature Extraction", "No sample found")
        return

    try:
        features = extract_features(sample)
        # Check dimensions
        # 13x2 (MFCC) + 2 (Centroid) + 2 (Rolloff) + 1 (Flatness) + 2 (Pitch) + 2 (ZCR) + 2 (RMSE) + 1 (Silence) + 1 (Smoothness) = 39
        expected_dim = 39 
        if len(features) == expected_dim:
            reporter.log_pass("Feature Vector Dimension")
        else:
            reporter.log_fail("Feature Vector Dimension", f"Expected {expected_dim}, got {len(features)}")
            
        if np.isnan(features).any():
             reporter.log_fail("NaN Check", "NaN values found in features")
        else:
             reporter.log_pass("NaN Check")
             
    except Exception as e:
        reporter.log_fail("Feature Extraction Crash", str(e))

def api_request(payload, headers=None):
    if headers is None:
        headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    try:
        return requests.post(API_URL, json=payload, headers=headers)
    except Exception as e:
        return None

def test_api_functional():
    print("\n--- PHASE 4: API FUNCTIONAL TESTS ---")
    
    # Prepare Data
    sample_file = get_audio_sample("human")
    if not sample_file:
         reporter.log_fail("API Tests", "No audio sample available for payload")
         return
         
    with open(sample_file, "rb") as f:
        valid_b64 = base64.b64encode(f.read()).decode()
        
    base_payload = {
        "language": "Tamil",
        "audioFormat": "mp3",
        "audioBase64": valid_b64
    }
    
    # 4.1 Valid Request
    resp = api_request(base_payload)
    if resp and resp.status_code == 200:
        data = resp.json()
        if data.get("status") == "success" and \
           data.get("classification") in ["HUMAN", "AI_GENERATED"] and \
           0.0 <= data.get("confidenceScore") <= 1.0 and \
           data.get("explanation"):
            reporter.log_pass("Valid Request Structure")
        else:
             reporter.log_fail("Valid Request Content", f"Invalid response body: {data}")
    else:
        reporter.log_fail("Valid Request Connection", f"Status {resp.status_code if resp else 'None'}")

    # 4.2 Language Coverage
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    all_langs_pass = True
    for lang in languages:
        payload = base_payload.copy()
        payload["language"] = lang
        resp = api_request(payload)
        if not resp or resp.status_code != 200:
            reporter.log_fail(f"Language Test {lang}", "Failed")
            all_langs_pass = False
    
    if all_langs_pass:
        reporter.log_pass("Language Coverage")

    # 4.3 Security Tests
    # Missing Key
    resp = requests.post(API_URL, json=base_payload) # No headers
    if resp.status_code == 403 or resp.status_code == 401 or resp.status_code == 422: # FastAPI depends checks usually return 403/422 if missing header completely
        reporter.log_pass("Security: Missing Key")
    else:
        reporter.log_fail("Security: Missing Key", f"Got {resp.status_code}")
        
    # Invalid Key
    resp = api_request(base_payload, headers={"x-api-key": "WRONG", "Content-Type": "application/json"})
    if resp.status_code == 401:
        reporter.log_pass("Security: Invalid Key")
    else:
        reporter.log_fail("Security: Invalid Key", f"Got {resp.status_code}")

    # 4.4 Input Validation
    # Invalid Base64
    bad_payload = base_payload.copy()
    bad_payload["audioBase64"] = "NotABase64String"
    resp = api_request(bad_payload)
    if resp.status_code == 400 or resp.status_code == 422: # 400 from our app, 422 from Pydantic
        reporter.log_pass("Input Validation: Bad Base64")
    else:
        reporter.log_fail("Input Validation: Bad Base64", f"Got {resp.status_code}")

def test_explanation_quality():
    print("\n--- PHASE 5: EXPLANATION QUALITY ---")
    # We already have a valid response from 4.1, let's just re-use logic
    sample_file = get_audio_sample("human") # Ideally check AI too
    if not sample_file: return
    
    with open(sample_file, "rb") as f:
        valid_b64 = base64.b64encode(f.read()).decode()
        
    payload = {"language": "English", "audioFormat": "mp3", "audioBase64": valid_b64}
    resp = api_request(payload)
    if resp and resp.status_code == 200:
        expl = resp.json().get("explanation", "")
        keywords = ["pitch", "jitter", "spectral", "breathing", "robotic", "natural"]
        if any(k in expl.lower() for k in keywords):
            reporter.log_pass("Explanation Relevance")
        else:
            reporter.log_warn("Explanation Relevance", f"Explanation '{expl}' might be generic")
            
def test_performance():
    print("\n--- PHASE 7: PERFORMANCE & STABILITY ---")
    sample_file = get_audio_sample("human")
    if not sample_file: return
    
    with open(sample_file, "rb") as f:
        valid_b64 = base64.b64encode(f.read()).decode()
    payload = {"language": "English", "audioFormat": "mp3", "audioBase64": valid_b64}
    
    success_count = 0
    start_time = time.time()
    
    for _ in range(20):
        resp = api_request(payload)
        if resp and resp.status_code == 200:
            success_count += 1
            
    duration = time.time() - start_time
    avg_lat = duration / 20
    
    if success_count == 20:
        reporter.log_pass(f"Stability (20/20 Success, Avg Latency: {avg_lat:.2f}s)")
    else:
        reporter.log_fail("Stability Test", f"Only {success_count}/20 passed")

if __name__ == "__main__":
    test_model_integrity()
    test_feature_extraction()
    # Ensure API is running for these
    try:
        requests.get("http://localhost:8001/")
        test_api_functional()
        test_explanation_quality()
        test_performance()
    except:
        reporter.log_fail("API Connection", "Server not reachable at http://localhost:8001. Start it first.")
    
    reporter.report()

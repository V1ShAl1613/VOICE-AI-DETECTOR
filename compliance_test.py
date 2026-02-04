"""
Comprehensive Compliance Test for Voice AI Detection API
Tests all requirements from the Problem Statement
"""
import requests
import base64
import json
import os
import time
import sys

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"
API_KEY = "secret123"
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

# Test results storage
results = []

def log_result(test_name, passed, details=""):
    status = "[PASS]" if passed else "[FAIL]"
    results.append({
        "test": test_name,
        "passed": passed,
        "details": details
    })
    print(f"{status} | {test_name}")
    if details and not passed:
        print(f"       Details: {details}")

def get_sample_audio_base64():
    """Get a valid MP3 audio file as base64 for testing"""
    dataset_path = "dataset"
    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.mp3'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'rb') as f:
                        return base64.b64encode(f.read()).decode('utf-8'), filepath
    return None, None

print("="*60)
print("VOICE AI DETECTION API - COMPLIANCE TEST REPORT")
print("="*60)
print(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Base URL: {BASE_URL}")
print("="*60)
print()

# ============================================
# SECTION 1: API AVAILABILITY
# ============================================
print("SECTION 1: API Availability")
print("-"*40)

try:
    resp = requests.get(f"{BASE_URL}/")
    log_result("API Root Endpoint", resp.status_code == 200, f"Status: {resp.status_code}")
except Exception as e:
    log_result("API Root Endpoint", False, str(e))

try:
    resp = requests.get(f"{BASE_URL}/docs")
    log_result("OpenAPI Documentation", resp.status_code == 200, f"Status: {resp.status_code}")
except Exception as e:
    log_result("OpenAPI Documentation", False, str(e))

print()

# ============================================
# SECTION 2: API AUTHENTICATION (Section 5)
# ============================================
print("SECTION 2: API Authentication (x-api-key)")
print("-"*40)

# Test missing API key
try:
    resp = requests.post(
        f"{BASE_URL}/api/voice-detection",
        headers={"Content-Type": "application/json"},
        json={"language": "Tamil", "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
    )
    log_result("Reject Missing API Key", resp.status_code == 401, f"Status: {resp.status_code}")
    
    # Check error format
    try:
        data = resp.json()
        has_status = "status" in str(data) and "error" in str(data)
        has_message = "message" in str(data)
        log_result("Error Response Format (status+message)", has_status and has_message, f"Response: {data}")
    except:
        log_result("Error Response Format (status+message)", False, "Invalid JSON")
except Exception as e:
    log_result("Reject Missing API Key", False, str(e))

# Test invalid API key
try:
    resp = requests.post(
        f"{BASE_URL}/api/voice-detection",
        headers={"Content-Type": "application/json", "x-api-key": "invalid_key"},
        json={"language": "Tamil", "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
    )
    log_result("Reject Invalid API Key", resp.status_code == 401, f"Status: {resp.status_code}")
except Exception as e:
    log_result("Reject Invalid API Key", False, str(e))

# Test valid API key
try:
    resp = requests.post(
        f"{BASE_URL}/api/voice-detection",
        headers=HEADERS,
        json={"language": "Tamil", "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
    )
    log_result("Accept Valid API Key", resp.status_code != 401, f"Status: {resp.status_code}")
except Exception as e:
    log_result("Accept Valid API Key", False, str(e))

print()

# ============================================
# SECTION 3: SUPPORTED LANGUAGES (Section 2)
# ============================================
print("SECTION 3: Supported Languages")
print("-"*40)

languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

audio_b64, audio_file = get_sample_audio_base64()
if audio_b64:
    print(f"   Using test file: {audio_file}")
    
    for lang in languages:
        try:
            resp = requests.post(
                f"{BASE_URL}/api/voice-detection",
                headers=HEADERS,
                json={"language": lang, "audioFormat": "mp3", "audioBase64": audio_b64}
            )
            passed = resp.status_code == 200
            if passed:
                data = resp.json()
                passed = data.get("language") == lang
            log_result(f"Language: {lang}", passed, f"Status: {resp.status_code}")
        except Exception as e:
            log_result(f"Language: {lang}", False, str(e))
else:
    print("   WARNING: No sample audio found - skipping language tests")
    for lang in languages:
        log_result(f"Language: {lang}", False, "No sample audio file")

# Test invalid language
try:
    resp = requests.post(
        f"{BASE_URL}/api/voice-detection",
        headers=HEADERS,
        json={"language": "French", "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
    )
    log_result("Reject Invalid Language (French)", resp.status_code == 422, f"Status: {resp.status_code}")
except Exception as e:
    log_result("Reject Invalid Language (French)", False, str(e))

print()

# ============================================
# SECTION 4: REQUEST FORMAT (Section 7)
# ============================================
print("SECTION 4: Request Format Validation")
print("-"*40)

# Missing language
try:
    resp = requests.post(
        f"{BASE_URL}/api/voice-detection",
        headers=HEADERS,
        json={"audioFormat": "mp3", "audioBase64": "dGVzdA=="}
    )
    log_result("Reject Missing Language Field", resp.status_code == 422, f"Status: {resp.status_code}")
except Exception as e:
    log_result("Reject Missing Language Field", False, str(e))

# Missing audioBase64
try:
    resp = requests.post(
        f"{BASE_URL}/api/voice-detection",
        headers=HEADERS,
        json={"language": "Tamil", "audioFormat": "mp3"}
    )
    log_result("Reject Missing audioBase64 Field", resp.status_code == 422, f"Status: {resp.status_code}")
except Exception as e:
    log_result("Reject Missing audioBase64 Field", False, str(e))

# audioFormat defaults to mp3
try:
    if audio_b64:
        resp = requests.post(
            f"{BASE_URL}/api/voice-detection",
            headers=HEADERS,
            json={"language": "Tamil", "audioBase64": audio_b64}
        )
        log_result("Default audioFormat to mp3", resp.status_code == 200, f"Status: {resp.status_code}")
    else:
        log_result("Default audioFormat to mp3", False, "No sample audio")
except Exception as e:
    log_result("Default audioFormat to mp3", False, str(e))

print()

# ============================================
# SECTION 5: RESPONSE FORMAT (Sections 8, 9)
# ============================================
print("SECTION 5: Response Format Validation")
print("-"*40)

if audio_b64:
    try:
        resp = requests.post(
            f"{BASE_URL}/api/voice-detection",
            headers=HEADERS,
            json={"language": "Tamil", "audioFormat": "mp3", "audioBase64": audio_b64}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            
            # Check all required fields
            log_result("Response has 'status' field", "status" in data, f"Value: {data.get('status')}")
            log_result("Response 'status' = 'success'", data.get("status") == "success", f"Value: {data.get('status')}")
            log_result("Response has 'language' field", "language" in data, f"Value: {data.get('language')}")
            log_result("Response has 'classification' field", "classification" in data, f"Value: {data.get('classification')}")
            log_result("Response has 'confidenceScore' field", "confidenceScore" in data, f"Value: {data.get('confidenceScore')}")
            log_result("Response has 'explanation' field", "explanation" in data, f"Value: {data.get('explanation')[:50] if data.get('explanation') else 'None'}...")
            
            # Classification values
            classification = data.get("classification")
            log_result("Classification is AI_GENERATED or HUMAN", 
                      classification in ["AI_GENERATED", "HUMAN"], 
                      f"Value: {classification}")
            
            # Confidence score range
            score = data.get("confidenceScore")
            log_result("confidenceScore in range [0.0, 1.0]", 
                      0.0 <= score <= 1.0 if score is not None else False,
                      f"Value: {score}")
            
            print(f"\n   Full Response:")
            print(f"   {json.dumps(data, indent=2)}")
        else:
            log_result("Success Response Format", False, f"Status: {resp.status_code}")
    except Exception as e:
        log_result("Response Format Validation", False, str(e))
else:
    print("   WARNING: No sample audio found - skipping response format tests")

print()

# ============================================
# SECTION 6: ERROR RESPONSE FORMAT (Section 11)
# ============================================
print("SECTION 6: Error Response Format")
print("-"*40)

try:
    resp = requests.post(
        f"{BASE_URL}/api/voice-detection",
        headers={"Content-Type": "application/json"},
        json={"language": "Tamil", "audioFormat": "mp3", "audioBase64": "dGVzdA=="}
    )
    
    data = resp.json()
    detail = data.get("detail", {})
    
    has_status = detail.get("status") == "error" if isinstance(detail, dict) else False
    has_message = "message" in detail if isinstance(detail, dict) else False
    
    log_result("Error has 'status': 'error'", has_status, f"Response: {detail}")
    log_result("Error has 'message' field", has_message, f"Response: {detail}")
except Exception as e:
    log_result("Error Response Format", False, str(e))

print()

# ============================================
# FINAL SUMMARY
# ============================================
print("="*60)
print("FINAL COMPLIANCE SUMMARY")
print("="*60)

passed_count = sum(1 for r in results if r["passed"])
failed_count = len(results) - passed_count
total_count = len(results)

print(f"Total Tests:  {total_count}")
print(f"Passed:       {passed_count}")
print(f"Failed:       {failed_count}")
print(f"Pass Rate:    {(passed_count/total_count*100):.1f}%")
print()

if failed_count > 0:
    print("Failed Tests:")
    for r in results:
        if not r["passed"]:
            print(f"  [FAIL] {r['test']}: {r['details']}")
    print()

if passed_count == total_count:
    print("ALL TESTS PASSED - API IS FULLY COMPLIANT!")
elif passed_count / total_count >= 0.9:
    print("MOSTLY COMPLIANT - Minor issues to address")
else:
    print("COMPLIANCE ISSUES - Please review failed tests")

print("="*60)

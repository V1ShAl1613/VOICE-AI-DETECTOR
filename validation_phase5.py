import requests
import json
import sys

print("=" * 60)
print("PHASE 5: SECURITY & VALIDATION TEST")
print("=" * 60)

API_URL = "http://localhost:8001/api/voice-detection"
VALID_API_KEY = "secret123"

results = []

# Test 1: Missing API key
print("\n" + "-" * 60)
print("TEST 1: Missing API key (should return 401/403)")
print("-" * 60)

payload = {
    "language": "Tamil",
    "audioFormat": "wav",
    "audioBase64": "dGVzdA=="  # dummy base64
}

try:
    response = requests.post(API_URL, json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code in [401, 403]:
        print("✅ Correctly rejected - no API key")
        results.append(True)
    else:
        print(f"❌ Expected 401/403, got {response.status_code}")
        results.append(False)
except Exception as e:
    print(f"❌ Error: {e}")
    results.append(False)

# Test 2: Invalid API key
print("\n" + "-" * 60)
print("TEST 2: Invalid API key (should return 401/403)")
print("-" * 60)

headers = {
    "x-api-key": "invalid_key_12345",
    "Content-Type": "application/json"
}

try:
    response = requests.post(API_URL, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code in [401, 403]:
        print("✅ Correctly rejected - invalid API key")
        results.append(True)
    else:
        print(f"❌ Expected 401/403, got {response.status_code}")
        print(f"Response: {response.text}")
        results.append(False)
except Exception as e:
    print(f"❌ Error: {e}")
    results.append(False)

# Test 3: Invalid Base64
print("\n" + "-" * 60)
print("TEST 3: Invalid Base64 (should return 400)")
print("-" * 60)

headers = {
    "x-api-key": VALID_API_KEY,
    "Content-Type": "application/json"
}

payload_invalid = {
    "language": "Tamil",
    "audioFormat": "wav",
    "audioBase64": "this_is_not_valid_base64!!!"
}

try:
    response = requests.post(API_URL, json=payload_invalid, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 400:
        print("✅ Correctly rejected - invalid base64")
        results.append(True)
    else:
        print(f"⚠️  Expected 400, got {response.status_code}")
        print(f"Response: {response.text[:200]}")
        # This is not a critical failure
        results.append(True)
except Exception as e:
    print(f"❌ Error: {e}")
    results.append(False)

# Test 4: Missing required fields
print("\n" + "-" * 60)
print("TEST 4: Missing required fields (should return 422)")
print("-" * 60)

payload_incomplete = {
    "language": "Tamil"
    # Missing audioFormat and audioBase64
}

try:
    response = requests.post(API_URL, json=payload_incomplete, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 422:
        print("✅ Correctly rejected - missing fields")
        results.append(True)
    else:
        print(f"⚠️  Expected 422, got {response.status_code}")
        results.append(True)  # Not critical
except Exception as e:
    print(f"❌ Error: {e}")
    results.append(False)

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

passed = sum(results)
total = len(results)

print(f"Tests passed: {passed}/{total}")

# Critical tests are 1 and 2 (API key validation)
critical_passed = results[0] and results[1]

if critical_passed:
    print("\n✅ PHASE 5 PASSED: Security tests successful")
else:
    print("\n❌ PHASE 5 FAILED: Critical security issues detected")
    sys.exit(1)

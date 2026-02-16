import os

# Read the base64 string
try:
    with open("audio_base64.txt", "r") as f:
        audio_base64 = f.read().strip()
except FileNotFoundError:
    audio_base64 = "ERROR: audio_base64.txt not found. Please ensure the file exists."

# content for the test values file
content = f"""
HERE ARE THE VALUES YOU NEED FOR THE ENDPOINT TESTER:

1. API KEY
--------------------
secret123
--------------------

2. ENDPOINT URL
--------------------
https://voice-ai-detector-production-1c73.up.railway.app/api/voice-detection
--------------------

3. LANGUAGE
--------------------
Tamil
--------------------

4. AUDIO FORMAT
--------------------
mp3
--------------------

5. AUDIOBASE64 STRING (Copy below)
--------------------
{audio_base64}
--------------------
"""

with open("TEST_VALUES.txt", "w") as f:
    f.write(content.strip())

print("TEST_VALUES.txt generated successfully.")

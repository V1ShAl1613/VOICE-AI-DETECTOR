"""
Start ngrok tunnel and keep it running
"""
from pyngrok import ngrok
import time

# Set auth token
ngrok.set_auth_token('39Bvw63ALSXYB2vX0NIH77vFsJI_2S8JD1jJj8xM5WBrcofds')

# Start tunnel
tunnel = ngrok.connect(8000)
print("="*60)
print("NGROK TUNNEL ACTIVE")
print("="*60)
print(f"Public URL: {tunnel.public_url}")
print(f"API Endpoint: {tunnel.public_url}/api/voice-detection")
print("="*60)
print("Keep this window open to maintain the tunnel!")
print("Press Ctrl+C to stop")
print("="*60)

# Keep the script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down tunnel...")
    ngrok.disconnect(tunnel.public_url)

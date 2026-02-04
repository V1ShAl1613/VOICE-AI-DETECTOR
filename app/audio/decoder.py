import base64
import tempfile
import os
import shutil
from fastapi import HTTPException

def decode_audio(base64_string: str) -> tuple[str, str]:
    """
    Decodes a base64 string and saves it to a temporary file.
    Returns the path to the temporary file and the temp directory.
    Uses directory-based temp strategy for Windows compatibility.
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(base64_string)
        
        # Create temporary directory (Windows-safe)
        tmp_dir = tempfile.mkdtemp(prefix="voice_input_")
        tmp_path = os.path.join(tmp_dir, "input.mp3")
        
        # Write audio to file
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
            
        return tmp_path, tmp_dir
    except Exception as e:
        raise HTTPException(status_code=400, detail={"status": "error", "message": f"Invalid value for 'audioBase64': {str(e)}"})

def cleanup_temp_dir(tmp_dir: str):
    """
    Removes the temporary directory and all contents.
    Safe for Windows - uses shutil.rmtree with error handler for file locking issues.
    """
    import stat
    
    if not tmp_dir or not os.path.exists(tmp_dir):
        return
    
    def handle_remove_readonly(func, path, exc):
        """Error handler for Windows readonly/locked files"""
        try:
            # Try to change permissions and retry
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except:
            # If still failing, just pass - temp will be cleaned by OS
            pass
    
    try:
        shutil.rmtree(tmp_dir, onerror=handle_remove_readonly)
    except Exception:
        # Silent fail - temp cleanup is best-effort
        pass


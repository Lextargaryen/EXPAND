import os
import sys
import shutil
import torch
from huggingface_hub import HfApi, login
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

def run_checks():
    print("🚀 Starting Pre-Flight Check for EXPAND...")
    print("-" * 40)
    
    # 1. Check Virtual Environment
    venv_path = "expand_env"
    if venv_path not in sys.prefix:
        print("❌ ERROR: You are NOT inside the 'expand_env' virtual environment.")
        print(f"   Run: source {os.path.join(os.getcwd(), venv_path, 'bin/activate')}")
    else:
        print("✅ Environment: expand_env is active.")

    # 2. Check GPU & CUDA
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA is not available. PyTorch cannot see your GPU.")
        print("   Check your NVIDIA drivers (nvidia-smi) and Arch Linux CUDA toolkit.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Detected: {gpu_name}")
        # Check VRAM (Llama 3 8B 4-bit needs ~5GB minimum for loading)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"📊 VRAM Available: {vram_gb:.2f} GB")

    # 3. Check Disk Space (Targeting the Hugging Face Cache)
    # We want at least 15GB free to be safe (model + temp files + outputs)
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    if free_gb < 15:
        print(f"⚠️ WARNING: Low disk space! Only {free_gb:.2f} GB free.")
    else:
        print(f"✅ Disk Space: {free_gb:.2f} GB free.")

    # 4. Check Hugging Face Authentication & Gating
    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"✅ Hugging Face: Logged in as {user_info['name']}")
        
        # Check if you actually have access to Llama 3 (The Gated Check)
        try:
            api.model_info("meta-llama/Meta-Llama-3-8B")
            print("✅ Llama 3 Access: Confirmed. You're on the list!")
        except GatedRepoError:
            print("❌ ERROR: You have not been granted access to Llama 3 yet.")
            print("   Go to: https://huggingface.co/meta-llama/Meta-Llama-3-8B and click 'Request Access'.")
        except Exception as e:
            print(f"⚠️ Could not verify Llama 3 access: {e}")

    except Exception:
        print("❌ ERROR: Not logged into Hugging Face.")
        print("   Running login prompt now...")
        login()

    # 5. Check Unsloth Installation
    try:
        import unsloth
        print("✅ Unsloth: Library is installed and ready.")
    except ImportError:
        print("❌ ERROR: Unsloth is not installed in this environment.")
        print("   Run: pip install unsloth")

    print("-" * 40)
    print("🏁 Pre-flight check complete!")

if __name__ == "__main__":
    run_checks()
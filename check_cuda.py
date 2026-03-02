"""
Advanced CUDA check for RTX 50-series and Blackwell architecture.
Run: python check_cuda.py
"""

import subprocess
import sys

def check_nvidia_smi():
    """Check CUDA via nvidia-smi (driver + GPU info)."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("=== NVIDIA Driver & GPU (nvidia-smi) ===")
            print(result.stdout)
            return True
        return False
    except (FileNotFoundError, Exception):
        print("nvidia-smi not found or error occurred.")
        return False

def check_pytorch_cuda():
    """Detailed check for PyTorch and GPU compatibility."""
    try:
        import torch
        print("=== PyTorch CUDA Details ===")
        print(f"PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            # Check architectures supported by this specific PyTorch build
            arch_list = torch.cuda.get_arch_list()
            device_name = torch.cuda.get_device_name(0)
            
            print(f"Device Name: {device_name}")
            print(f"PyTorch Built for Architectures: {', '.join(arch_list)}")
            
            # --- THE ACTUAL TEST ---
            print("\n--- Running Tensor Operation Test ---")
            try:
                # This will fail on your machine until you update PyTorch
                x = torch.rand(3, 3).cuda()
                y = x @ x
                print("Test Success: Tensor operation completed on GPU!")
                return True
            except Exception as e:
                print(f"Test FAILED: {e}")
                if "sm_120" in str(e) or "capability" in str(e).lower():
                    print("\n[!] DIAGNOSIS: Your RTX 5060 Ti (sm_120) is too new for this PyTorch build.")
                    print("[!] ACTION: Install a newer PyTorch version (likely a Nightly/Preview build).")
                return False
        else:
            print("PyTorch cannot see any CUDA device.")
            return False
            
    except ImportError:
        print("PyTorch not installed.")
        return None

def main():
    print("CUDA Compatibility Check (Blackwell Edition)\n")
    check_nvidia_smi()
    print()
    torch_working = check_pytorch_cuda()
    
    print("\n" + "="*40)
    if torch_working:
        print("Result: EVERYTHING OK. Your 5060 Ti is ready for AI.")
    else:
        print("Result: ACTION REQUIRED. Check the PyTorch install steps below.")
        print("Try: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")
    print("="*40)

if __name__ == "__main__":
    main()
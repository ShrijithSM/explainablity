"""
Fix for torchvision/PyTorch compatibility issues

This script fixes the "operator torchvision::nms does not exist" error
by updating package versions.
"""

import subprocess
import sys

def fix_torchvision():
    """Fix torchvision compatibility with PyTorch."""
    print("=" * 70)
    print("FIXING TORCHVISION COMPATIBILITY")
    print("=" * 70)
    
    print("\n[1/2] Uninstalling incompatible torchvision...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], check=True)
        print("✓ Torchvision uninstalled")
    except:
        print("⚠ Torchvision not found (this is OK)")
    
    print("\n[2/2] Installing compatible torchvision...")
    try:
        # Install torchvision compatible with PyTorch 2.0+
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torchvision>=0.15.0", "--upgrade"
        ], check=True)
        print("✓ Compatible torchvision installed")
    except Exception as e:
        print(f"✗ Failed: {e}")
        print("\nTry manually:")
        print("  pip install --upgrade torch torchvision")
        return False
    
    print("\n" + "=" * 70)
    print("SUCCESS! Try running your experiment again:")
    print("  python experiments/exp1_correlational_mapping.py")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = fix_torchvision()
    exit(0 if success else 1)

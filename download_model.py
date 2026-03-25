"""
Model Downloader for Chronoscope

Downloads the required model from HuggingFace to local cache.
Run this ONCE before running experiments.

Usage:
    python download_model.py
    python download_model.py --model gpt2  # For a smaller model
"""

import argparse
import os
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name="Qwen/Qwen2.5-0.5B"):
    """Download model and tokenizer to HuggingFace cache."""
    print(f"=" * 70)
    print(f"Downloading model: {model_name}")
    print(f"This may take a few minutes depending on your internet speed...")
    print(f"=" * 70)
    
    # Temporarily disable offline mode
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    
    try:
        # Download model
        print(f"\n[1/2] Downloading model files...")
        snapshot_path = snapshot_download(
            repo_id=model_name,
            local_files_only=False,
        )
        print(f"✓ Model downloaded to: {snapshot_path}")
        
        # Download tokenizer
        print(f"\n[2/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        print(f"✓ Tokenizer downloaded")
        
        print(f"\n" + "=" * 70)
        print(f"SUCCESS! Model '{model_name}' is now cached locally.")
        print(f"You can now run experiments:")
        print(f"  python experiments/exp1_correlational_mapping.py")
        print(f"=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n" + "=" * 70)
        print(f"ERROR: Failed to download model")
        print(f"Reason: {str(e)}")
        print(f"\nPossible solutions:")
        print(f"1. Check your internet connection")
        print(f"2. Try a smaller model: python download_model.py --model gpt2")
        print(f"3. Use a model you already have cached")
        print(f"=" * 70)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models for Chronoscope")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model name from HuggingFace (default: Qwen/Qwen2.5-0.5B)"
    )
    
    args = parser.parse_args()
    
    print("\n🤖 CHRONOSCOPE MODEL DOWNLOADER\n")
    
    success = download_model(args.model)
    
    if success:
        exit(0)
    else:
        exit(1)

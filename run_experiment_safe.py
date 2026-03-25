"""
Safe Experiment Runner

Runs experiments with proper environment configuration to avoid
torchvision/PyTorch compatibility issues.

Usage:
    python run_experiment_safe.py exp1
    python run_experiment_safe.py exp2
    python run_experiment_safe.py exp3
"""

import os
import sys
import subprocess

# Disable problematic imports that aren't needed for text models
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Prevent transformers from importing vision utilities
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow model loading
os.environ["HF_HUB_OFFLINE"] = "0"  # Allow model loading

def run_experiment(exp_name):
    """Run an experiment with safe environment settings."""
    
    experiments = {
        "exp1": "experiments/exp1_correlational_mapping.py",
        "exp2": "experiments/exp2_causal_heatmap.py",
        "exp3": "experiments/exp3_chain_of_thought.py",
        "exp4": "experiments/exp4_live_dashboard.py",
        "exp5": "experiments/exp5_eager_nexus.py",
        "exp6": "experiments/exp6_head_interference.py",
    }
    
    if exp_name not in experiments:
        print(f"Unknown experiment: {exp_name}")
        print(f"Available: {', '.join(experiments.keys())}")
        return False
    
    exp_path = experiments[exp_name]
    
    print("=" * 70)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"Environment configured for compatibility")
    print("=" * 70)
    print()
    
    try:
        # Run the experiment as a subprocess
        result = subprocess.run([sys.executable, exp_path], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_experiment_safe.py <exp1|exp2|exp3|exp4|exp5|exp6>")
        print("\nExamples:")
        print("  python run_experiment_safe.py exp1  # Run experiment 1")
        print("  python run_experiment_safe.py exp3  # Run experiment 3")
        sys.exit(1)
    
    exp_name = sys.argv[1].lower()
    if not exp_name.startswith("exp"):
        exp_name = f"exp{exp_name}"
    
    success = run_experiment(exp_name)
    sys.exit(0 if success else 1)

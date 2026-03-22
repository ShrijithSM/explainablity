"""
Chronoscope CLI — Run interpretability experiments.

Usage:
    python run_experiment.py --experiment exp1
    python run_experiment.py --experiment exp1 --model Qwen/Qwen2.5-0.5B
    python run_experiment.py --experiment exp1 --model deepseek-ai/deepseek-coder-1.3b-base
    python run_experiment.py --experiment exp1 --prompt "If x > 5 and x < 10, then x is"
    python run_experiment.py --experiment exp1 --load-4bit
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chronoscope.config import ChronoscopeConfig


def main():
    parser = argparse.ArgumentParser(
        description="Chronoscope: Glass-Box Observability for LLM Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --experiment exp1
  python run_experiment.py --experiment exp1 --model Qwen/Qwen2.5-1.5B
  python run_experiment.py --experiment exp1 --prompt "2+2="
  python run_experiment.py --list-layers --model Qwen/Qwen2.5-0.5B
        """,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["exp1", "exp2", "exp3", "exp5", "exp6"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name or local path. Defaults to config.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override the default reasoning prompt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load model in 4-bit quantization (saves VRAM).",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=8,
        help="Number of SVD components for dimensionality reduction.",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Just list hookable layers and exit.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution.",
    )

    args = parser.parse_args()

    # Build config from CLI args
    config = ChronoscopeConfig(
        load_in_4bit=args.load_4bit,
        max_new_tokens=args.max_tokens,
        svd_components=args.svd_components,
    )

    if args.model:
        config.model_name = args.model

    if args.cpu:
        config.device = "cpu"

    if args.prompt:
        config.clean_prompt = args.prompt

    # List layers mode
    if args.list_layers:
        from chronoscope.models import load_model, list_hookable_layers

        model, _ = load_model(config)
        list_hookable_layers(model)
        return

    # Run experiment
    if args.experiment == "exp1":
        from experiments.exp1_correlational_mapping import run

        run(config)
    elif args.experiment == "exp2":
        from experiments.exp2_causal_heatmap import run

        run(config)
    elif args.experiment == "exp3":
        from experiments.exp3_chain_of_thought import run

        run(config)
    elif args.experiment == "exp5":
        from experiments.exp5_eager_nexus import run

        import asyncio
        asyncio.run(run(config))
    elif args.experiment == "exp6":
        from experiments.exp6_head_interference import run

        import asyncio
        asyncio.run(run(config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

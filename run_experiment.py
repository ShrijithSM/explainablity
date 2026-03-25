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
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Loop continuously through prompts, keeping model + dashboard alive (exp1 only).",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to a text file with one prompt per line (used with --continuous).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="Seconds to pause between prompts in continuous mode (default: 2).",
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

        if getattr(args, "continuous", False):
            # Delegate entirely to exp1's __main__ continuous logic by
            # reconstructing argv and re-running the module's if-block inline.
            import itertools, time as _time
            from experiments.exp1_correlational_mapping import (
                _PROMPT_BANK,
                ChronoscopeInterceptor,
                DashboardBridge,
            )
            from chronoscope.models import load_model as _lm
            from rich.console import Console as _C
            _con = _C()

            _prompts_list = list(_PROMPT_BANK)
            if getattr(args, "prompts_file", None):
                with open(args.prompts_file, "r", encoding="utf-8") as _pf:
                    _prompts_list = [l.strip() for l in _pf if l.strip()]
            elif args.prompt:
                _prompts_list = [args.prompt]

            _con.rule("[bold magenta]Chronoscope — Continuous Mode[/]")
            _con.print(f"  Prompts in rotation: {len(_prompts_list)}")
            _con.print("  Press [bold]Ctrl+C[/] to stop.\n")

            _m, _tok = _lm(config)
            _icp = ChronoscopeInterceptor(_m, _tok, config)
            _br = None
            try:
                _br = DashboardBridge(
                    transport=getattr(config, "dashboard_transport", "websocket"),
                    ws_port=int(getattr(config, "dashboard_ws_port", 8765)),
                ).start()
                _url = _br.serve_dashboard(
                    getattr(config, "dashboard_html_path", "integration_hub/frontend/public/chronoscope_live.html"),
                    port=int(getattr(config, "dashboard_http_port", 8766)),
                )
                _con.print(f"  [bold green]Dashboard:[/] {_url}")
            except Exception as _be:
                _con.print(f"  [yellow]Dashboard bridge disabled: {_be}[/]")

            _it = 1
            try:
                for _pr in itertools.cycle(_prompts_list):
                    _con.rule(f"[cyan]Iteration {_it}[/]")
                    config.clean_prompt = _pr
                    run(config, _model=_m, _tokenizer=_tok, _interceptor=_icp, _bridge=_br)
                    _it += 1
                    _pause = getattr(args, "pause", 2.0)
                    if _pause > 0:
                        _time.sleep(_pause)
            except KeyboardInterrupt:
                _con.print("\n[yellow]Continuous mode stopped.[/]")
            finally:
                if _br:
                    _br.stop()
                _icp.cleanup()
        else:
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

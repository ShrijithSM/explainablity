# Chronoscope Developer Log

## 2026-03-06

### Major Updates
- **Fixed 4-bit Loading Error**: Resolved `ValueError: .to is not supported` by disabling quantization for the 0.5B model. For small models, `bitsandbytes` + `accelerate` conflicts are avoided by loading in full/half precision (still fits easily in 6GB VRAM).
- **Exp2 Live Dashboard**: Ported the real-time visualization logic from `exp4` to `exp2` (`exp2_live_heatmap.py`). 
    - Added a live-updating `rich.Table` heatmap.
    - Added an interactive loop to allow multiple analysis runs without restarting the script.
    - Added `-p` / `--prompt` CLI support to collect input before initializing the heavy TUI.
- **Random Patching Noise**: Set the default `patching_noise` in `config.py` to `gaussian` per user request ("keep it random").
- **Terminal Optimization**: Addressed terminal "spam" by keeping a single persistent process for live experiments and using `Stop-Process` to clean up stale GPU/Python instances.

### Technical Decisions
- **Quantization Policy**: For models < 1B parameters, Chronoscope defaults to FP16/BF16 on 6GB+ cards to maximize stability. Quantization is reserved for 7B+ models or ultra-low VRAM scenarios.
- **UI Flow**: Prompt collection is moved *before* model loading. This minimizes the "frozen" state of the terminal during startup.
- **Divergence Metric**: Using normalized L2 distance for the heatmap, with intensity-based color mapping (White → Blue → Green → Yellow → Red).

### Ongoing Tasks
- [ ] Add graceful error handling for out-of-memory (OOM) during long generation traces.
- [ ] Implement incremental SVD update if trajectory length exceeds `max_cache_size`.

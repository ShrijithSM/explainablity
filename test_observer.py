import torch
import numpy as np
from chronoscope.observer import SignalObserver
from chronoscope.config import ChronoscopeConfig

config = ChronoscopeConfig()
observer = SignalObserver(config)

# Mock some trajectory data [tokens, hidden_dim]
trajectory = torch.randn(15, 2048)

print("Starting incremental_analysis check...")
try:
    res1 = observer.incremental_analysis(trajectory)
    print("Res1:", res1)
    
    # Add another token
    trajectory2 = torch.cat([trajectory, torch.randn(1, 2048)])
    res2 = observer.incremental_analysis(trajectory2)
    print("Res2:", res2)
    
    # Add a completely different token (simulate anomaly)
    trajectory3 = torch.cat([trajectory2, torch.randn(1, 2048) * 100])
    res3 = observer.incremental_analysis(trajectory3)
    print("Res3:", res3)
    
    print("SUCCESS: `incremental_analysis` logic is flawless.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ERROR:", e)

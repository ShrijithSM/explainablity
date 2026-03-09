"""
Chronoscope: A Glass-Box Observability Engine for LLM Reasoning Traces.

Treats LLM hidden states as multivariate time series and applies classical
signal processing + causal inference to mathematically verify reasoning validity.
"""

__version__ = "0.1.0"

from .config import ChronoscopeConfig
from .models import load_model, list_hookable_layers
from .interceptor import ChronoscopeInterceptor
from .observer import SignalObserver
from .analyzer import CausalAnalyzer
from .synthesizer import ReportSynthesizer

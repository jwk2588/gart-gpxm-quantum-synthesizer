"""GART v3.0 + GPXM Quantum Linguistic Synthesizer.

A dual-swarm AI music production system combining:
- GART: Generative Adversarial Rap Tournament
- GPXM: Genetic Persona & Expansive Memory framework
- LLLM: Latent Linguistic Lil Model (PyTorch cross-attention)
- Stash Box reconstruction engine (Tay B + Lil Durk)
- Diagram-to-Code Inversion Agent
- Adaptive Resilience (Moore's Law vs Murphy's Law)
"""

__version__ = "3.0.0"
__author__ = "GART-GPXM Team"
__license__ = "MIT"

# Lazy imports for heavy modules
def _lazy_import(name):
    import importlib
    return importlib.import_module(f"gart_gpxm.{name}")

# Version info
VERSION = __version__
VERSION_TUPLE = tuple(map(int, __version__.split(".")))

# Module availability
MODULES = {
    "core": ["lllm_architecture", "daw_utilities", "dual_swarm_orchestrator"],
    "gpxm": ["genetic_persona", "entropic_scripting", "roster_manager"],
    "inversion": ["diagram_to_code_agent"],
    "resilience": ["adaptive_engine"],
    "tournament": ["tournament_engine"],
    "stash_box": ["reconstruction_engine"],
    "cross_platform": ["adapter"],
}

def get_module_status():
    """Return availability status of all modules."""
    status = {}
    for pkg, modules in MODULES.items():
        for mod in modules:
            try:
                _lazy_import(f"{pkg}.{mod}")
                status[f"{pkg}.{mod}"] = "available"
            except ImportError as e:
                status[f"{pkg}.{mod}"] = f"unavailable: {e}"
    return status

__all__ = [
    "__version__",
    "VERSION",
    "VERSION_TUPLE",
    "MODULES",
    "get_module_status",
]
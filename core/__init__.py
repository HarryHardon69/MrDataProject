# Sophia_Alpha2/core/__init__.py
"""
Core package for Sophia_Alpha2 Cognitive Architecture.
This file makes 'core' a Python package and can be used to expose 
key components of the core modules, making them easily importable.
"""
import sys
import os

# Specific imports from the top-level 'config' package for use in this __init__.py
try:
    from config import VERBOSE_OUTPUT, PERSONA_NAME, SYSTEM_LOG_PATH
except ModuleNotFoundError:
    # Fallback if 'config' package isn't immediately on sys.path.
    # This is less likely to be needed here if main.py correctly sets up sys.path,
    # but good for robustness if core package is somehow accessed differently.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config import VERBOSE_OUTPUT, PERSONA_NAME, SYSTEM_LOG_PATH

# Import key functions/classes from core modules to make them available 
# when 'core' is imported, or to define the package's public API via __all__.

from .brain import think, get_shared_manifold, SpacetimeManifold
from .dialogue import generate_response, dialogue_loop, get_dialogue_persona
from .ethics import score_ethics, track_trends
from .gui import start_gui 
from .library import (
    sanitize_text, 
    summarize_text, 
    is_valid_coordinate,
    Mitigator,
    CoreException, BrainError, PersonaError, MemoryError, EthicsError, DialogueError
)
from .memory import (
    calculate_novelty, 
    store_memory, 
    get_memory_by_id, 
    get_memories_by_concept_name,
    get_recent_memories,
    read_memory 
)
from .persona import Persona

__all__ = [
    'think', 'get_shared_manifold', 'SpacetimeManifold',
    'generate_response', 'dialogue_loop', 'get_dialogue_persona',
    'score_ethics', 'track_trends',
    'start_gui',
    'sanitize_text', 'summarize_text', 'is_valid_coordinate', 'Mitigator',
    'CoreException', 'BrainError', 'PersonaError', 'MemoryError', 'EthicsError', 'DialogueError',
    'calculate_novelty', 'store_memory', 'get_memory_by_id', 
    'get_memories_by_concept_name', 'get_recent_memories', 'read_memory',
    'Persona',
]

# Use the imported VERBOSE_OUTPUT, PERSONA_NAME, SYSTEM_LOG_PATH directly
if VERBOSE_OUTPUT: # No need for hasattr(config, ...) if imported directly
    print("Core package initialized. Modules exposed:", sorted(__all__))
    print(f"  Configuration loaded. Example: PERSONA_NAME = {PERSONA_NAME}") # Use imported PERSONA_NAME
    if SYSTEM_LOG_PATH: # Check if it has a value
        print(f"  System log path: {SYSTEM_LOG_PATH}") # Use imported SYSTEM_LOG_PATH
    else:
        print("  System log path: Not configured.")

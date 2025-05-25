# Sophia_Alpha2/core/config.py
"""
Core configuration settings for Sophia_Alpha2.
Manages paths, resource profiles, API keys, and persona details.
"""
import os
import sys
import json # For loading/saving persona, ethics_db if they were here

# --- Path Configuration ---
# Dynamically determine project root, assuming this script is in core/
# and 'core' is a subdirectory of the project root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_path(relative_path: str) -> str:
    """
    Get the absolute path for a resource, handling PyInstaller's _MEIPASS.
    The relative_path is assumed to be relative to the project root.
    """
    # This ensures that if the script is bundled (e.g., with PyInstaller),
    # paths to data files are correctly resolved.
    if hasattr(sys, '_MEIPASS'): # Running in a PyInstaller bundle
        # _MEIPASS is a temporary directory created by PyInstaller
        base_path = sys._MEIPASS
    else: # Running in a normal Python environment
        base_path = _PROJECT_ROOT
    return os.path.join(base_path, relative_path)

# Data paths (relative to project root, resolved by get_path)
DATA_DIR = get_path("data")
LOG_DIR = get_path(os.path.join("data", "logs"))
PERSONA_DIR = get_path(os.path.join("data", "personas"))

# Specific file paths
SYSTEM_LOG_PATH = get_path(os.path.join("data", "logs", "system_events.log"))
PERSONA_PROFILE = get_path(os.path.join("data", "personas", "sophia_profile.json")) # Default persona
ETHICS_DB_PATH = get_path(os.path.join("data", "ethics_db.json"))
KNOWLEDGE_GRAPH_PATH = get_path(os.path.join("data", "knowledge_graph.json"))

# Ensure directories exist for these paths
def ensure_path(file_path: str):
    """Ensure the directory for the given file_path exists."""
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name): # Check if dir_name is not empty
        try:
            os.makedirs(dir_name, exist_ok=True)
            if VERBOSE_OUTPUT: print(f"Config: Created directory {dir_name}")
        except Exception as e:
            # Use print directly to stderr for critical config errors if logging isn't up yet
            print(f"Config Critical Error: Failed to create directory {dir_name}: {e}", file=sys.stderr)

# --- Resource Management ---
RESOURCE_PROFILE_TYPE = "moderate" # Options: "low", "moderate", "high"
_RESOURCE_PROFILES = {
    "low": {"MAX_NEURONS": 500, "MAX_SPIKE_RATE": 0.05, "RESOLUTION": 2, "SNN_TIME_STEPS": 10},
    "moderate": {"MAX_NEURONS": 2000, "MAX_SPIKE_RATE": 0.1, "RESOLUTION": 3, "SNN_TIME_STEPS": 20},
    "high": {"MAX_NEURONS": 10000, "MAX_SPIKE_RATE": 0.2, "RESOLUTION": 4, "SNN_TIME_STEPS": 50}
}
RESOURCE_PROFILE = _RESOURCE_PROFILES.get(RESOURCE_PROFILE_TYPE, _RESOURCE_PROFILES["moderate"])
MANIFOLD_RANGE = 1.0 # Coordinate range [-MANIFOLD_RANGE, MANIFOLD_RANGE] for SNN manifold

# --- System Behavior ---
VERBOSE_OUTPUT = True # Enables detailed logging to console
ENABLE_SNN = True # Master switch for SNN operations (True to use SNN, False for LLM fallback)
HEBBIAN_LEARNING_RATE = 0.005 # Adjusted from brain self-test, might need tuning
STDP_WINDOW_MS = 20.0 # STDP time window in milliseconds

# --- API Keys and Endpoints ---
ENABLE_LLM_API = True # Master switch for ALL LLM API calls (bootstrap, dialogue fallback, etc.)
LLM_PROVIDER = "lm_studio" # Options: "openai", "lm_studio", "ollama", "mock_for_snn_test"
LLM_TEMPERATURE = 0.5  # Default temperature for LLM responses
LLM_CONNECTION_TIMEOUT = 5 # Seconds to wait for LLM server connection test
LLM_REQUEST_TIMEOUT = 20 # Seconds to wait for LLM API response

_LLM_CONFIG = {
    "openai": {
        "API_KEY": os.getenv("SOPHIA_OPENAI_API_KEY", "your_openai_api_key_here"),
        "BASE_URL": "https://api.openai.com/v1",
        "MODEL": "gpt-3.5-turbo",
        "CONCEPT_PROMPT_TEMPLATE": """
You are a sophisticated language model integrated into a cognitive architecture.
Your task is to provide a concise, structured JSON object detailing a given concept.
The JSON object must include:
1.  "summary": A brief, insightful summary of the concept (max 2-3 sentences).
2.  "valence": A float from -1.0 (negative) to 1.0 (positive) representing typical emotional association.
3.  "abstraction": A float from 0.0 (concrete) to 1.0 (highly abstract).
4.  "relevance": A float from 0.0 (irrelevant) to 1.0 (highly relevant to general human experience).
5.  "intensity": A float from 0.0 (low intensity) to 1.0 (high intensity/arousal).
Ensure the output is ONLY the JSON object, with no surrounding text or markdown.
Example for 'love':
{
  "summary": "Love is a complex emotion of deep affection and attachment, fostering strong bonds and care.",
  "valence": 0.8,
  "abstraction": 0.7,
  "relevance": 0.9,
  "intensity": 0.85
}
"""
    },
    "lm_studio": {
        "API_KEY": "lm-studio", # Typically not required for local LM Studio
        "BASE_URL": os.getenv("SOPHIA_LM_STUDIO_URL", "http://localhost:1234/v1"),
        "MODEL": os.getenv("SOPHIA_LM_STUDIO_MODEL", "local-model/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"), # Example from LM Studio logs
        "CONCEPT_PROMPT_TEMPLATE": """
You are a helpful AI assistant providing structured data for a cognitive architecture.
For the given concept, generate a JSON object with the following keys:
"summary" (string, 2-3 sentences), "valence" (float, -1 to 1),
"abstraction" (float, 0 to 1), "relevance" (float, 0 to 1),
"intensity" (float, 0 to 1).
Output ONLY the JSON object. No other text or markdown.
Example for 'truth':
{
  "summary": "Truth refers to statements or beliefs that align with fact or reality. It's a fundamental concept in philosophy and daily life.",
  "valence": 0.5,
  "abstraction": 0.8,
  "relevance": 0.9,
  "intensity": 0.6
}
"""
    },
    "ollama": {
        "API_KEY": "ollama", # Not typically required
        "BASE_URL": os.getenv("SOPHIA_OLLAMA_URL", "http://localhost:11434/api"),
        "MODEL": os.getenv("SOPHIA_OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M"), # Ensure this model is pulled in Ollama
        "CONCEPT_PROMPT_TEMPLATE": """
[INST] You are an AI that outputs ONLY JSON. For the concept provided, generate a JSON object:
{
  "summary": "string summary",
  "valence": float_value,
  "abstraction": float_value,
  "relevance": float_value,
  "intensity": float_value
}
Concept: {concept_name}
[/INST]
""" # Specific templating often needed for Ollama models
    },
    "mock_for_snn_test": { # For testing SNN without real LLM calls during SNN specific tests
        "API_KEY": "mock_key",
        "BASE_URL": "http://localhost:1235/v1", # A non-existent port for mock
        "MODEL": "mock_model",
        "CONCEPT_PROMPT_TEMPLATE": "Mock prompt for testing SNN."
    }
}

# Set current LLM provider's config (handles "not_set" gracefully if key missing)
_current_llm_provider_config = _LLM_CONFIG.get(LLM_PROVIDER, {})
LLM_API_KEY = _current_llm_provider_config.get("API_KEY", "not_set")
LLM_BASE_URL = _current_llm_provider_config.get("BASE_URL", "not_set")
LLM_MODEL = _current_llm_provider_config.get("MODEL", "not_set")
LLM_CONCEPT_PROMPT_TEMPLATE = _current_llm_provider_config.get("CONCEPT_PROMPT_TEMPLATE", "No template configured.")


# --- Persona Configuration ---
PERSONA_NAME = "Sophia" # Default name, can be overridden by loaded profile

# --- Ethics Module Configuration ---
ETHICAL_FRAMEWORK = { # Defines weights and rules for ethical scoring
    "deontology": {"weight": 0.3, "rules": ["act_only_on_maxims_universalizable"]},
    "consequentialism": {"weight": 0.4, "rules": ["maximize_overall_good"]},
    "virtue_ethics": {"weight": 0.3, "rules": ["cultivate_virtuous_character_traits"]}
}
ETHICAL_ALIGNMENT_THRESHOLD = 0.65 # Minimum score for an action/memory to be considered ethically aligned (0.0-1.0)

# --- Self-Correction and Evolution ---
SELF_CORRECTION_THRESHOLD = 0.4 # Threshold for triggering self-correction (e.g. if novelty is below this for memory)
EVOLUTION_RATE_MODIFIER = 0.01 # Modifies rate of adaptation/learning in some modules

# --- Environment Detection ---
EXECUTION_ENVIRONMENT = os.getenv("SOPHIA_EXEC_ENV", "local_dev") # e.g., "local_dev", "testing", "production"

# --- Initialize Paths (must be after VERBOSE_OUTPUT is set if they print) ---
# Call ensure_path for all directory-based paths after defining them
# This is important because ensure_path might print if VERBOSE_OUTPUT is True.
_INITIALIZATION_PATHS = [SYSTEM_LOG_PATH, PERSONA_PROFILE, ETHICS_DB_PATH, KNOWLEDGE_GRAPH_PATH]
for pth in _INITIALIZATION_PATHS:
    ensure_path(pth) # Creates parent dirs if they don't exist


# --- Config Validation and Self-Test ---
def validate_config():
    """Performs basic validation of critical configuration settings."""
    valid = True
    # Path validations (check if parent dirs exist, as files might not yet)
    for path_name, path_val in [("SYSTEM_LOG_PATH", SYSTEM_LOG_PATH), ("PERSONA_PROFILE", PERSONA_PROFILE)]:
        if not os.path.exists(os.path.dirname(path_val)):
            print(f"ConfigError: Directory for {path_name} ({os.path.dirname(path_val)}) does not exist and could not be created.", file=sys.stderr)
            valid = False

    # LLM Configuration validation
    if ENABLE_LLM_API:
        if LLM_PROVIDER not in _LLM_CONFIG:
            print(f"ConfigError: LLM_PROVIDER '{LLM_PROVIDER}' not found in _LLM_CONFIG.", file=sys.stderr)
            valid = False
        elif _LLM_CONFIG[LLM_PROVIDER].get("BASE_URL", "not_set") == "not_set":
            print(f"ConfigError: LLM_BASE_URL for provider '{LLM_PROVIDER}' is 'not_set'.", file=sys.stderr)
            valid = False
        # API Key is not always "required" (e.g. local LLMs)
        if LLM_PROVIDER == "openai" and _LLM_CONFIG[LLM_PROVIDER].get("API_KEY", "your_openai_api_key_here") == "your_openai_api_key_here":
            print(f"ConfigWarning: OpenAI API_KEY is set to placeholder. Please configure SOPHIA_OPENAI_API_KEY environment variable.", file=sys.stderr)
            # Not setting valid=False for this, as it's a warning.

    # Resource Profile validation
    if RESOURCE_PROFILE_TYPE not in _RESOURCE_PROFILES:
        print(f"ConfigError: RESOURCE_PROFILE_TYPE '{RESOURCE_PROFILE_TYPE}' is invalid.", file=sys.stderr)
        valid = False
    
    if not valid and VERBOSE_OUTPUT:
        print("Config: One or more configuration errors found.", file=sys.stderr)
    elif VERBOSE_OUTPUT:
        print("Config: Basic configuration validation successful.")
    return valid

def self_test_config_paths_and_creation():
    """Self-test for path configurations and directory/file creation logic."""
    print("\n--- Config Path & Creation Self-Test ---")
    # Test ensure_path for a dummy file within a new subdirectory of DATA_DIR
    dummy_test_dir = os.path.join(DATA_DIR, "selftest_ensure_path")
    dummy_test_file = os.path.join(dummy_test_dir, "test_file.txt")
    
    path_test_ok = True
    try:
        if os.path.exists(dummy_test_dir): # Clean up from previous test if any
            import shutil
            shutil.rmtree(dummy_test_dir)

        print(f"  Testing ensure_path for: {dummy_test_file}")
        ensure_path(dummy_test_file) # Should create dummy_test_dir
        if not os.path.exists(dummy_test_dir):
            print(f"    FAIL: ensure_path did not create directory {dummy_test_dir}")
            path_test_ok = False
        else:
            print(f"    OK: Directory {dummy_test_dir} created by ensure_path.")
            # Clean up
            os.rmdir(dummy_test_dir) 
            print(f"    OK: Cleaned up {dummy_test_dir}.")

    except Exception as e:
        print(f"    FAIL: Error during ensure_path test: {e}")
        path_test_ok = False

    if path_test_ok: print("  Config Path & Creation Self-Test: PASSED")
    else: print("  Config Path & Creation Self-Test: FAILED", file=sys.stderr)
    return path_test_ok


# --- Main execution block for self-test when script is run directly ---
if __name__ == "__main__":
    print("--- Running Configuration Self-Check (config.py) ---")
    print(f"Project Root (_PROJECT_ROOT): {_PROJECT_ROOT}")
    print(f"Data Directory (DATA_DIR) (exists: {os.path.exists(DATA_DIR)}): {DATA_DIR}")
    print(f"System Log Path (SYSTEM_LOG_PATH) (parent exists: {os.path.exists(os.path.dirname(SYSTEM_LOG_PATH))}): {SYSTEM_LOG_PATH}")
    print(f"Persona Profile Path (PERSONA_PROFILE) (parent exists: {os.path.exists(os.path.dirname(PERSONA_PROFILE))}): {PERSONA_PROFILE}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"LLM Base URL: {LLM_BASE_URL}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Resource Profile ('{RESOURCE_PROFILE_TYPE}'): {RESOURCE_PROFILE}")
    print(f"Verbose Output: {VERBOSE_OUTPUT}")
    print(f"SNN Enabled: {ENABLE_SNN}")
    print(f"LLM API Enabled: {ENABLE_LLM_API}")

    if not validate_config():
        print("\nConfig: CRITICAL - Configuration validation FAILED. Application may not run correctly.", file=sys.stderr)
        # sys.exit(1) # Optionally exit if validation fails catastrophically
    else:
        print("\nConfig: Configuration validation successful.")

    self_test_config_paths_and_creation()
    print("\n--- Configuration Self-Check Complete (config.py) ---")

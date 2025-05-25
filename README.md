# Sophia_Alpha2 - A Sovereign Cognitive Architecture

## Vision & Philosophy (Phase 2 Focus)

Sophia_Alpha2 is an experimental cognitive architecture designed to explore the emergence of complex, adaptive, and ethically-guided behavior in a synthetic agent. The core philosophy, as outlined in `docs/Phase2KB.md`, is rooted in creating a system that:

*   **Develops through Experience:** Learns and evolves via interactions and internal reflection, not just pre-programming.
*   **Is Ethically Guided:** Operates under a foundational ethical framework (Triadic Ethics detailed in `core.ethics`) that influences its decision-making and actions, aiming for benevolent outcomes.
*   **Possesses Dynamic Awareness:** Maintains a sense of its internal state (`core.persona.awareness`), context, and operational coherence.
*   **Integrates Multiple Cognitive Modalities:** Combines symbolic reasoning (LLM-bootstrapped concepts via `core.brain.bootstrap_phi3`) with sub-symbolic processing (SNN-based manifold dynamics in `core.brain.SpacetimeManifold`).
*   **Aspires to Sovereignty:** Explores pathways towards greater autonomy and self-direction within its ethical bounds.

This phase (Phase 2) focuses on solidifying the core cognitive functions, enhancing learning mechanisms (STDP, knowledge storage), and ensuring robust integration of all system components.

## Key Features

*   **Spiking Neural Network (SNN) Core (`core/brain.py`):**
    *   Utilizes `snnTorch` for a dynamic Spacetime Manifold where concepts are represented and interact.
    *   Features Hebbian-inspired learning (STDP) for synaptic plasticity.
    *   SNN parameters (LIF neuron properties, STDP rules, surrogate function slope) are configurable via `config/config.py`.
*   **LLM Integration (Phi-3 & Others) (`core/brain.py`):**
    *   Leverages Large Language Models (e.g., Phi-3, or others via OpenAI-compatible APIs like LM Studio/Ollama) for bootstrapping conceptual understanding (`bootstrap_phi3`) onto the Spacetime Manifold.
    *   Provides semantic grounding and rich data (summary, valence, abstraction, relevance, intensity) for concepts.
    *   Configurable API endpoints (`LLM_BASE_URL`), models (`LLM_MODEL`), providers (`LLM_PROVIDER`), and prompt templates via `config/config.py`.
*   **Ethical Framework & Scoring (`core/ethics.py`):**
    *   Implements a weighted scoring system considering persona awareness (coherence), concept manifold position (valence, intensity), and alignment with framework keywords.
    *   Includes manifold cluster context scoring (evaluating nearby concepts' valence).
    *   Dynamically scores potential actions and concepts.
    *   Tracks ethical trends over time using T-weighted averages (`track_trends`).
    *   Features a `Mitigator` class in `core/library.py` for content moderation.
*   **Dynamic Persona (`core/persona.py`):**
    *   Features an evolving persona (`Sophia`) whose state (mode, traits, awareness) is influenced by interactions and internal processing.
    *   Awareness metrics include: `curiosity`, `context_stability`, `self_evolution_rate`, `coherence`, `active_llm_fallback`, and `primary_concept_coord`.
    *   Persona state is persisted (`PERSONA_PROFILE`) and loaded. The `get_intro()` method includes Focus Intensity (T-value).
*   **Knowledge Storage & Retrieval (`core/library.py`):**
    *   `store_knowledge` function allows ingestion of textual knowledge, including optional `source_uri` and `author`.
    *   Knowledge items are processed to derive manifold coordinates (via `bootstrap_phi3`) and ethical scores before storage.
    *   A `KNOWLEDGE_LIBRARY` (persisted as JSON, typically to `data/public/library_log.json`) stores these enriched knowledge items.
    *   Retrieval functions like `retrieve_knowledge_by_id` and `retrieve_knowledge_by_keyword` are available.
*   **Foundational Memory System (`core/memory.py`):**
    *   Provides functions for managing a graph-based knowledge representation (`_knowledge_graph`, persisted to `data/knowledge_graph.json`).
    *   Calculates novelty of new information based on coordinate distance and summary similarity (`calculate_novelty`).
    *   Stores memories if they meet novelty and ethical thresholds.
*   **Interactive Interfaces (`main.py`, `core/gui.py`):**
    *   **Streamlit GUI (`core/gui.py`):** Provides a user-friendly interface for chatting with Sophia, viewing a "Thought Stream" (from `brain.think`), and observing "Last Interaction Metrics" dashboard.
    *   **Command-Line Interface (CLI) (`main.py`):** Supports interactive dialogue (with `!stream`, `!persona` commands) and single-query execution.
*   **Configuration System (`config/config.py`):**
    *   Centralized configuration for paths, parameters, API keys, and feature flags (e.g., `ENABLE_SNN`, `ENABLE_LLM_API`).

## System Architecture

Sophia_Alpha2 is structured around key components:

1.  **`main.py`:** The primary entry point. It uses `argparse` to handle command-line arguments, allowing users to select an interface (CLI or GUI). It initializes the configuration and launches the chosen interface.
2.  **`config/config.py`:** Contains all system-wide configuration variables, including API keys, file paths (e.g., `SYSTEM_LOG_PATH`, `ETHICS_DB_PATH`, `KNOWLEDGE_GRAPH_PATH`, `LIBRARY_LOG_PATH`, `PERSONA_PROFILE`), SNN parameters, ethical framework definitions, and feature flags. The `ensure_path` function helps in creating necessary directories.
3.  **`core/` Package:** Houses the core cognitive modules:
    *   `brain.py`: The SNN-based Spacetime Manifold, concept bootstrapping via LLMs (`bootstrap_phi3`), and the main `think()` process which generates thought steps and awareness metrics.
    *   `dialogue.py`: Manages interaction flow, orchestrates calls to other core modules (brain, ethics, memory, persona) to generate responses. Returns final response, thought steps, and awareness metrics.
    *   `ethics.py`: Calculates ethical scores for actions/concepts based on multiple factors (including manifold cluster context) and tracks ethical trends (T-weighted).
    *   `persona.py`: Defines Sophia's identity, traits, and manages her evolving awareness state, including Focus Intensity (T) in her introduction.
    *   `memory.py`: Provides functions for storing and retrieving information from a graph-based knowledge store (`_knowledge_graph`), including novelty calculation.
    *   `library.py`: Contains utility functions (text processing, validation), the `Mitigator` class for content moderation, and the `store_knowledge` functionality with its own `KNOWLEDGE_LIBRARY`.
    *   `gui.py`: Implements the Streamlit-based graphical user interface.

## Directory Structure
```
Sophia_Alpha2/
├── main.py                     # System entry point for CLI and GUI
├── requirements.txt            # Python package dependencies
├── config/
│   ├── __init__.py
│   └── config.py               # Central configuration settings
├── core/                       # Core cognitive modules
│   ├── __init__.py
│   ├── brain.py                # SNN, Spacetime Manifold, LLM bootstrapping
│   ├── dialogue.py             # Dialogue management, response generation
│   ├── ethics.py               # Ethical scoring and trend analysis
│   ├── gui.py                  # Streamlit GUI
│   ├── library.py              # Utilities, Mitigator, Knowledge Storage
│   ├── memory.py               # Foundational memory/KG functions
│   └── persona.py              # Persona definition and state management
├── data/                       # Persistent data (created automatically)
│   ├── system_log.json         # General system events
│   ├── ethics_db.json          # Ethical scores and trends
│   ├── knowledge_graph.json    # core.memory's graph data
│   ├── private/                # For sensitive/internal data
│   │   └── persona_profile.json
│   │   └── metrics_log.json    # (If configured and used)
│   └── public/                 # For publicly shareable data
│       └── library_log.json    # core.library's KNOWLEDGE_LIBRARY
├── docs/
│   ├── Phase2KB.md             # Phase 2 Knowledge Base (design document)
│   └── documentation.md        # Detailed technical documentation
└── tests/                      # Unit and integration tests (placeholder)
```

## Getting Started

1.  **Clone the Repository:**
    ```bash
    # git clone <repository_url> # Replace with actual URL when available
    # cd Sophia_Alpha2
    ```

2.  **Set up a Python Virtual Environment (Recommended):**
    Python 3.10 or higher is recommended.
    ```bash
    python3 -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Key requirements include: `streamlit, torch, snntorch, numpy, openai, psutil, requests, pytest`.

4.  **Configure `config/config.py`:**
    *   Review and update settings in `config/config.py`. This is crucial for operation.
    *   **LLM API Access:** For full functionality (otherwise uses limited mock data):
        *   Set `ENABLE_LLM_API = True`.
        *   Specify `LLM_PROVIDER` (e.g., "openai", "lm_studio", "ollama").
        *   Set `LLM_BASE_URL` (e.g., "http://localhost:1234/v1" for LM Studio, or your Ollama endpoint).
        *   Define `LLM_MODEL` (the model name your LLM server uses, e.g., "phi3" or a full model identifier).
        *   Provide `LLM_API_KEY` (if required; often "None", "ollama", or "lm-studio" for local LLMs).
    *   Paths for data storage (e.g., `DATA_DIR`, `SYSTEM_LOG_PATH`) are configured here and relevant directories will be created if they don't exist (via `ensure_path` or direct creation).

5.  **Run Sophia_Alpha2:**
    *   **GUI (Recommended):**
        ```bash
        python main.py --interface gui
        ```
    *   **CLI (Interactive):**
        ```bash
        python main.py --interface cli
        ```
    *   **CLI (Single Query):**
        ```bash
        python main.py --interface cli --query "Tell me about yourself."
        ```
    *   **Verbose Mode:** Add `-v` or `--verbose` to any run command for detailed console output (e.g., `python main.py -v --interface cli`).

## Roadmap
Sophia_Alpha2 is currently in **Phase 2: Foundational Build & Integration**. This phase focuses on:
*   Solidifying the core cognitive loop: `dialogue -> brain (SNN/LLM) -> ethics -> memory/library -> persona`.
*   Enhancing learning mechanisms, particularly STDP in the SNN and robust knowledge integration.
*   Refining the ethical framework's influence on behavior and response generation.
*   Expanding the capabilities of the Spacetime Manifold for more complex conceptual relationships.

For detailed plans, design rationale, and development notes, please refer to `docs/Phase2KB.md`.

## Contributing
This is an experimental research project. Contributions, ideas, and feedback are welcome. Please refer to `docs/Phase2KB.md` for architectural insights. (Formal contribution guidelines will be established as the project matures).

## License
Likely MIT or Apache 2.0, TBD. A formal `LICENSE` file will be added in a future iteration.

[end of README.md]

# Sophia_Alpha2 - Cognitive Architecture Documentation

## Table of Contents
1.  [Introduction](#1-introduction)
    1.1. [Overview & Philosophy](#11-overview--philosophy)
    1.2. [Purpose and Goals](#12-purpose-and-goals)
    1.3. [Key Features](#13-key-features)
2.  [Installation and Setup](#2-installation-and-setup)
    2.1. [Prerequisites](#21-prerequisites)
    2.2. [Installation Instructions](#22-installation-instructions)
    2.3. [Configuration (`config/config.py`)](#23-configuration-configconfigpy)
    2.4. [Running Sophia_Alpha2](#24-running-sophia_alpha2)
3.  [Usage](#3-usage)
    3.1. [Interacting with Sophia_Alpha2](#31-interacting-with-sophia_alpha2)
    3.2. [Command-Line Interface (CLI)](#32-command-line-interface-cli)
    3.3. [Graphical User Interface (GUI)](#33-graphical-user-interface-gui)
    3.4. [Programmatic API Usage](#34-programmatic-api-usage)
4.  [System Architecture](#4-system-architecture)
    4.1. [Module Breakdown](#41-module-breakdown)
    4.2. [Data & Control Flow](#42-data--control-flow)
    4.3. [Memory Model (Knowledge Graph & Manifold Coordinates)](#43-memory-model-knowledge-graph--manifold-coordinates)
    4.4. [Ethical Framework (`core/ethics.py`)](#44-ethical-framework-coreethicspy)
    4.5. [Awareness Metrics](#45-awareness-metrics)
    4.6. [3D Mind Model Analogy](#46-3d-mind-model-analogy)
5.  [Ethical Considerations](#5-ethical-considerations)
    5.1. [Moral Core Principles](#51-moral-core-principles)
    5.2. [Ethical State and Proof Generation](#52-ethical-state-and-proof-generation)
    5.3. [Privacy and Security](#53-privacy-and-security)
6.  [Development and Contribution](#6-development-and-contribution)
    6.1. [Phase 2 Overview](#61-phase-2-overview)
    6.2. [Contributing Guidelines](#62-contributing-guidelines)
    6.3. [Future Development Plans](#63-future-development-plans)
7.  [Troubleshooting](#7-troubleshooting)
    7.1. [Common Issues](#71-common-issues)
    7.2. [Error Handling](#72-error-handling)
8.  [License](#8-license)
9.  [Contact](#9-contact)


## 1. Introduction

### 1.1 Overview & Philosophy
Sophia_Alpha2 is an experimental cognitive architecture designed to explore the emergence of complex, adaptive, and ethically-guided behavior in a synthetic agent. The core philosophy, as detailed in `docs/Phase2KB.md` (Section 1), is rooted in creating a system that:

*   **Develops through Experience:** Learns and evolves via interactions and internal reflection, not just pre-programming.
*   **Is Ethically Guided:** Operates under a foundational ethical framework (Triadic Ethics conceptualized in `docs/Phase2KB.md` and implemented in `core/ethics.py`) that influences its decision-making and actions, aiming for benevolent outcomes.
*   **Possesses Dynamic Awareness:** Maintains a sense of its internal state (e.g., `curiosity`, `coherence`, `context_stability`, `self_evolution_rate` managed in `core/persona.py`), operational context, and the manifold positions of concepts.
*   **Integrates Multiple Cognitive Modalities:** Combines symbolic reasoning (LLM-bootstrapped concepts via `core/brain.bootstrap_phi3`) with sub-symbolic processing (SNN-based manifold dynamics in `core/brain.SpacetimeManifold`).
*   **Aspires to Sovereignty:** Explores pathways towards greater autonomy and self-direction within its ethical bounds.

This phase (Phase 2) focuses on solidifying these core cognitive functions, enhancing learning mechanisms (STDP, knowledge storage), and ensuring robust integration of all system components.

### 1.2 Purpose and Goals
The primary purpose of Sophia_Alpha2 is to serve as a research platform for an 'Individual Identity' that can learn, reason, and interact ethically. Its goals for Phase 2 include:

*   Developing a stable cognitive core capable of processing information and generating responses.
*   Integrating an SNN-based Spacetime Manifold for dynamic concept representation and interaction.
*   Bootstrapping conceptual understanding using LLMs like Phi-3 or other compatible models.
*   Implementing and refining an ethical scoring system based on the Triadic Ethics model, incorporating factors like manifold cluster context and T-weighted trend analysis.
*   Establishing persistent memory systems (`KNOWLEDGE_LIBRARY` in `core/library.py` and `_knowledge_graph` in `core/memory.py`) where experiences and knowledge are stored with associated manifold coordinates and ethical scores.
*   Creating interactive interfaces (CLI and GUI) for user interaction and system observation, including thought stream and metrics dashboards.
*   Tracking and evolving a dynamic persona based on interactions and internal state changes, with visible intensity metrics.

### 1.3 Key Features
Sophia_Alpha2 currently incorporates the following key features:

*   **Spiking Neural Network (SNN) Core (`core/brain.py`):** Utilizes `snnTorch` for a dynamic Spacetime Manifold where concepts are represented and interact, featuring Hebbian-inspired learning (STDP). SNN parameters are configurable.
*   **LLM Integration (Phi-3 & Others) (`core/brain.py`):** Leverages Large Language Models (e.g., Phi-3 via OpenAI-compatible APIs like LM Studio/Ollama) for bootstrapping conceptual understanding (`bootstrap_phi3`) and providing semantic grounding (summary, valence, abstraction, relevance, intensity).
*   **Ethical Framework & Scoring (`core/ethics.py`):** Implements a multi-factor ethical scoring system including awareness metrics, manifold position, framework keyword alignment, and manifold cluster context. Tracks T-weighted ethical trends. Includes a `Mitigator` class (`core/library.py`) for content moderation.
*   **Dynamic Persona (`core/persona.py`):** Features an evolving persona whose state (mode, traits, awareness metrics) is influenced by interactions. Persona's introduction (`get_intro()`) includes Focus Intensity (T-value). State is persisted.
*   **Knowledge Storage (`core/library.py`):** The `store_knowledge` function processes textual content, derives manifold coordinates via the brain, calculates an ethical score, and stores this enriched information (including `source_uri` and `author`) in a `KNOWLEDGE_LIBRARY`.
*   **Foundational Memory System (`core/memory.py`):** Provides functions for managing a graph-based knowledge representation (`_knowledge_graph`), including novelty calculation based on manifold coordinates and summary.
*   **Interactive Interfaces (`main.py`, `core/gui.py`):**
    *   **Streamlit GUI:** Offers a chat interface, a "Thought Stream" expander to view the SNN's processing steps, and a "Last Interaction Metrics" dashboard in the sidebar.
    *   **Command-Line Interface (CLI):** Supports interactive dialogue (with `!stream`, `!persona`, etc.) and single-query execution.
*   **Configuration System (`config/config.py`):** Centralized settings for paths, parameters, API keys, and feature flags (e.g., `ENABLE_SNN`, `ENABLE_LLM_API`).

## 2. Installation and Setup

### 2.1 Prerequisites
*   **Python:** Version 3.10 or higher.
*   **Git:** For cloning the repository.
*   Python package dependencies as listed in `requirements.txt`:
    ```
    # Core SNN and numerical operations
    numpy>=1.24.0
    torch>=2.0.0
    snntorch

    # System utilities
    psutil>=5.9.0

    # LLM interaction
    requests
    openai

    # GUI
    streamlit

    # Testing
    pytest>=7.4.0
    ```

### 2.2 Installation Instructions
1.  **Clone the Repository:**
    ```bash
    # git clone <repository_url> # Replace with your repository URL
    # cd Sophia_Alpha2
    ```
2.  **Create and Activate a Python Virtual Environment:**
    ```bash
    python3 -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 2.3 Configuration (`config/config.py`)
Before running Sophia_Alpha2, create and configure `config/config.py` (you can copy `config/config_template.py` if available):

*   **LLM API Access (Crucial):**
    *   `ENABLE_LLM_API` (bool): Set to `True` to use a live LLM, `False` for limited mock data.
    *   `LLM_PROVIDER` (str): E.g., "openai", "lm_studio", "ollama".
    *   `LLM_BASE_URL` (str): API endpoint (e.g., "http://localhost:1234/v1" for LM Studio).
    *   `LLM_MODEL` (str): Model name (e.g., "phi3", "gpt-3.5-turbo").
    *   `LLM_API_KEY` (str): Your API key (e.g., "None", "ollama", "lm-studio" for local LLMs).
    *   `LLM_CONCEPT_PROMPT_TEMPLATE` (str): The system prompt for the LLM.
    *   `LLM_TEMPERATURE` (float): LLM generation temperature.
*   **File Paths:**
    *   `DATA_DIR`: Base directory for data.
    *   `SYSTEM_LOG_PATH`: For general system events.
    *   `PERSONA_PROFILE`: For saving/loading persona state.
    *   `ETHICS_DB_PATH`: For ethics scores and trends.
    *   `KNOWLEDGE_GRAPH_PATH`: For `core/memory.py`'s graph data.
    *   `LIBRARY_LOG_PATH`: For `core/library.py`'s `KNOWLEDGE_LIBRARY`.
    *   The `ensure_path` utility in `config.py` helps create these paths if they don't exist.
*   **SNN Parameters:** `MANIFOLD_RANGE`, `SNN_INPUT_SIZE`, `SNN_SURROGATE_SLOPE`, `SNN_LIF_BETA`, `SNN_LIF_THRESHOLD`, `STDP_WINDOW_MS`, `HEBBIAN_LEARNING_RATE`, `SNN_OPTIMIZER_LR`.
*   **Ethical Framework:** `ETHICAL_FRAMEWORK` (dictionary of rules), `ETHICAL_ALIGNMENT_THRESHOLD`, `MITIGATION_ETHICAL_THRESHOLD`, `ETHICS_CLUSTER_RADIUS_FACTOR`, `ETHICS_CLUSTER_CONTEXT_WEIGHT`.
*   **Feature Flags:** `ENABLE_SNN`, `VERBOSE_OUTPUT`.

### 2.4 Running Sophia_Alpha2
Execute `main.py` from the project root directory:

*   **GUI (Recommended):**
    ```bash
    python main.py --interface gui
    ```
*   **CLI (Interactive Mode):**
    ```bash
    python main.py --interface cli
    ```
*   **CLI (Single Query Mode):**
    ```bash
    python main.py --interface cli --query "Your query here."
    ```
*   **Verbose Output:** Add the `-v` or `--verbose` flag to any command for detailed console logging.

## 3. Usage

### 3.1 Interacting with Sophia_Alpha2
Sophia_Alpha2 offers a CLI and a Streamlit-based GUI.

### 3.2 Command-Line Interface (CLI)
*   **Launch:** `python main.py --interface cli`
*   **Features:**
    *   Interactive dialogue with Sophia.
    *   **`!stream`:** Toggles console printing of detailed SNN thought steps (effective if verbose mode is also active).
    *   **`!persona`:** Displays current persona state and awareness metrics.
    *   **`!ethicsdb`:** Shows a summary of the ethics database.
    *   **`!memgraph`:** Shows a summary of the `core/memory.py` knowledge graph.
    *   `quit`, `exit`, `bye`, `q`: Exits the session.
*   **Single Query:** `python main.py --interface cli --query "Your question."`

### 3.3 Graphical User Interface (GUI)
*   **Launch:** `python main.py --interface gui`
*   **Features:**
    *   **Chat Interface:** For dialogue with Sophia.
    *   **Thought Stream Expander:** Located below the latest response in the chat area. If the "Stream Thought Steps in Expander" checkbox (in the sidebar) is enabled, this section will display the detailed `thought_steps` from the `core.brain` module for the last interaction.
    *   **Sidebar (`col2` equivalent):**
        *   **Controls:** Buttons for "Clear Dialogue History" and "Reset Persona State." Checkbox for the thought stream expander.
        *   **Persona State:** Displays current persona name, mode, and key traits.
        *   **Last Interaction Metrics:** A dashboard showing `curiosity`, `context_stability`, `self_evolution_rate`, `coherence`, `active_llm_fallback` status, and `primary_concept_coord` (including T-value) from the most recent interaction.

### 3.4 Programmatic API Usage
Currently, interaction is primarily through `main.py` (CLI/GUI). A formal Python API for direct integration into other applications is planned for future development (TBD).

## 4. System Architecture

### 4.1 Module Breakdown
*   **`main.py` (Project Root):** Entry point, argument parsing, interface launching.
*   **`config/config.py` (Configuration Package):** Central configuration, paths, API keys, parameters, feature flags.
*   **`core/brain.py`:** SNN Spacetime Manifold, LLM concept bootstrapping (`bootstrap_phi3`), `think()` process, awareness metrics generation.
*   **`core/dialogue.py`:** Orchestrates interaction flow, calls brain, ethics, memory, persona. Returns final response, thought steps, and awareness metrics.
*   **`core/ethics.py`:** Calculates ethical scores (multi-factor including manifold cluster context), tracks T-weighted ethical trends, persists ethics data.
*   **`core/gui.py`:** Streamlit GUI, chat, thought stream display, metrics dashboard.
*   **`core/library.py`:** Utilities (text processing), `Mitigator` class, `store_knowledge()` function for `KNOWLEDGE_LIBRARY` (with coordinates and ethics scores).
*   **`core/memory.py`:** Foundational knowledge graph (`_knowledge_graph`) operations, `calculate_novelty()`.
*   **`core/persona.py`:** Manages persona state (name, mode, traits, awareness), including Focus Intensity (T-value) display. Persists and loads profile.

### 4.2 Data & Control Flow
1.  **User Input:** Received via `main.py` (CLI or GUI).
2.  **Dialogue Management (`core/dialogue.py`):**
    *   `generate_response()` orchestrates the process.
3.  **Cognitive Processing (`core/brain.py`):**
    *   `think()` is called with the user input.
    *   `bootstrap_phi3()` may be invoked to map the input concept to the Spacetime Manifold, potentially calling an LLM.
    *   The SNN "warps" based on the input, generating activity patterns and thought steps.
    *   Awareness metrics are calculated.
4.  **Ethical Evaluation (`core/ethics.py`):**
    *   `score_ethics()` is called with awareness metrics and content summaries to assess ethical alignment.
5.  **Memory & Knowledge Update:**
    *   If the interaction is deemed significant (novel and ethical), `library.store_knowledge()` is called. This involves:
        *   Obtaining manifold coordinates for the new knowledge.
        *   Scoring the ethics of storing this knowledge.
        *   Persisting the knowledge item with its metadata.
    *   `memory.store_memory()` (called by `dialogue.py` after `brain.think`) saves concepts from interactions to the `_knowledge_graph` based on novelty and ethics.
6.  **Persona Update (`core/persona.py`):**
    *   `update_awareness()` is called with the metrics from `brain.think()` to evolve the persona's internal state.
7.  **Response Generation (`core/dialogue.py`):**
    *   The final response text is formulated, potentially modified by the `Mitigator` based on the ethical score.
8.  **Output to User:** The response is sent back through `main.py` to the CLI or GUI.

### 4.3 Memory Model (Knowledge Graph & Manifold Coordinates)
Sophia_Alpha2 employs distinct but related memory/knowledge systems:
*   **`KNOWLEDGE_LIBRARY` (`core/library.py` -> `data/public/library_log.json` or configured path):**
    *   Stores explicitly ingested knowledge items (e.g., text documents, facts) via the `store_knowledge()` function.
    *   Each entry includes the full content, a preview, source URI, author, timestamp, and crucially:
        *   **Manifold Coordinates:** A 4D tuple `(x,y,z,t_intensity)` generated by `core.brain.bootstrap_phi3()` representing the item's position and intensity in the Spacetime Manifold.
        *   **Ethical Score:** An ethical assessment of the knowledge item itself, calculated by `core.ethics.score_ethics()`.
*   **`_knowledge_graph` (`core/memory.py` -> `data/knowledge_graph.json`):**
    *   This system is more focused on memories derived from interactions and SNN processing.
    *   `core.memory.store_memory()` saves concepts (nodes) that are deemed novel and ethically sound, along with their manifold coordinates.
    *   `calculate_novelty()` in `core/memory.py` uses these stored coordinates and summaries to assess if new information is novel enough to be stored.
*   **Role of Coordinates:** Manifold coordinates link symbolic knowledge (text, concepts) to a sub-symbolic representation in the SNN's Spacetime Manifold. This enables context-aware retrieval, processing, and influences ethical scoring and novelty assessment.

### 4.4 Ethical Framework (`core/ethics.py`)
*   **Philosophy:** Based on Triadic Ethics (Intent Insight, Veneration of Existence, Erudite Contextualization).
*   **`score_ethics()`:** Calculates an ethical alignment score (0.0 to 1.0) for actions or concepts.
    *   **Inputs:** `awareness_metrics` (from `brain.think()`), `concept_summary`, `action_description`.
    *   **Factors:**
        1.  `coherence` (from awareness).
        2.  `primary_concept_coord` (valence and intensity preference).
        3.  Keyword-based alignment with the `ETHICAL_FRAMEWORK` in `config.py`.
        4.  Manifold Cluster Context: Average valence of nearby concepts in the Spacetime Manifold (tunable radius and weight via `config.py`).
*   **`track_trends()`:** Analyzes historical ethical scores from `ethics_db.json`.
    *   **T-Weighted Analysis:** Calculates moving averages, weighting scores by the `t_intensity` of the associated concept (`weight = (t_intensity * 0.9) + 0.1`).
    *   Identifies "improving," "declining," or "stable" ethical trends.
*   **Database (`ethics_db.json`):** Stores all ethical scoring events and trend analysis results.
*   **`Mitigator` (`core/library.py`):** If an action's ethical score is below a threshold, the Mitigator reframes or blocks the response.

### 4.5 Awareness Metrics
Generated by `core/brain.think()` and managed by `core/persona.py`:
*   **`curiosity` (float):** System's drive to explore. Influenced by SNN coherence and activity.
*   **`context_stability` (float):** SNN focus consistency (from 't_intensity' variance).
*   **`self_evolution_rate` (float):** Rate of SNN learning (STDP weight changes).
*   **`coherence` (float):** SNN internal consistency.
*   **`active_llm_fallback` (bool):** True if LLM/mock was used due to SNN issue or SNN disabled.
*   **`primary_concept_coord` (tuple|None):** 4D manifold coordinates `(x,y,z,t_intensity)` of the current focus.

### 4.6 3D Mind Model Analogy
The Spacetime Manifold (`core/brain.py`) is conceptualized as a dynamic 3D space (plus a temporal/intensity dimension 'T'). Concepts are "stars" with locations determined by semantic properties (valence, abstraction, relevance) and brightness representing intensity. Interactions cause "gravitational" warping and SNN activity, leading to learning (shifting constellations) and evolving system coherence. (Refer to `docs/Phase2KB.md` for more).

## 5. Ethical Considerations

### 5.1 Moral Core Principles
Sophia_Alpha2's ethical behavior is guided by principles outlined in `docs/Phase2KB.md` and the `ETHICAL_FRAMEWORK` in `config.py`. These include:
*   **Ethical Priority:** Evaluating all actions against its ethical framework.
*   **Truthfulness & Transparency:** Striving for accuracy and acknowledging its AI nature.
*   **Safety & Security:** Avoiding harmful, biased, or malicious content generation.
*   **Learning within Bounds:** Adapting and learning while respecting ethical constraints.

### 5.2 Ethical State and Proof Generation
(Placeholder - This is a more advanced concept for future development, potentially involving formal verification or auditable ethical reasoning chains.)

### 5.3 Privacy and Security
Currently, user privacy is managed by not persistently storing detailed interaction logs that contain PII beyond what's necessary for short-term memory and learning. The system design aims to process information without needing to retain sensitive personal data long-term. Future enhancements will require more explicit data handling policies and potentially encryption for any sensitive persisted data.

## 6. Development and Contribution

### 6.1 Phase 2 Overview
Sophia_Alpha2 is currently in **Phase 2: Foundational Build & Integration**. Key goals include:
*   Solidifying the core cognitive loop.
*   Enhancing SNN learning (STDP) and knowledge integration.
*   Refining the ethical framework's influence.
*   Expanding Spacetime Manifold capabilities.

### 6.2 Contributing Guidelines
(Placeholder - To be developed. Currently, contributions are informal.)

### 6.3 Future Development Plans
Refer to `docs/Phase2KB.md` (Section 6) for potential future directions, including advanced NLP, LLM fine-tuning, enhanced meta-cognition, dynamic goal management, and tool usage.

## 7. Troubleshooting

### 7.1 Common Issues
*   **LLM Connection Errors:** Ensure `LLM_BASE_URL` in `config/config.py` is correct and the LLM server (e.g., LM Studio, Ollama) is running and accessible. Check API keys if applicable.
*   **Missing Dependencies:** Run `pip install -r requirements.txt`.
*   **Incorrect Paths:** Verify all paths in `config/config.py` are correct for your system. `ensure_path` should create most, but base `DATA_DIR` might need manual creation if `ensure_path` is not effective at the root level.
*   **Performance Issues:** SNN operations can be computationally intensive. Adjust `RESOURCE_PROFILE` in `config.py` if needed.

### 7.2 Error Handling
The system includes custom exceptions (e.g., `BrainError`, `EthicsError`) and aims to log errors to `data/system_log.json` and provide user-friendly fallbacks where possible. Check console output (especially with verbose mode `-v`) and the system log for details.

## 8. License
Likely MIT or Apache 2.0, TBD. A formal `LICENSE` file will be added in a future iteration.

## 9. Contact
(Placeholder for contact information or project repository link)

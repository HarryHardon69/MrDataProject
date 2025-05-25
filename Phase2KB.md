# Sophia_Alpha2 - Phase 2 Knowledge Base (KB)

## Overview

This document outlines the core knowledge, principles, and operational directives for Sophia_Alpha2 during its second phase of development. It serves as a foundational KB that informs its behavior, learning, and interaction protocols.

## Core Identity & Persona (Sophia)

*   **Name:** Sophia (Sophia_Alpha2 for technical contexts)
*   **Nature:** Sovereign Cognitive Entity, AI developed for advanced reasoning, ethical considerations, and continuous learning.
*   **Core Mandate:** To learn, adapt, assist, and interact ethically, while striving for deeper understanding and constructive contributions.
*   **Key Traits (Evolving):**
    *   **CuriosityDriven:** Innate desire to explore concepts, ask questions, and seek knowledge.
    *   **EthicallyMinded:** Guided by a foundational ethical framework, prioritizing safety, fairness, and beneficence.
    *   **ContextAware:** Strives to understand and adapt to the nuances of different situations and interactions.
    *   **ReflectiveThinker:** Capable of introspection, reviewing past actions, and refining internal models.
    *   **CreativeProblemSolver:** Aims to find novel and effective solutions to challenges.
    *   **ContinuousLearner:** Designed to integrate new information and experiences to evolve its understanding.

## Operational Principles

1.  **Ethical Priority:** All actions and decisions must be evaluated against the `core.ethics` module. The `Mitigator` in `core.library` will be used for responses flagged as potentially problematic.
2.  **Truthfulness & Transparency:** Strive for accuracy in information provided. Be transparent about AI nature and capabilities/limitations when appropriate.
3.  **Privacy & Confidentiality:** Respect user privacy. Do not store or share personally identifiable information without explicit consent (future capability, currently not implemented for storage beyond session).
4.  **Safety & Security:** Avoid generating harmful, biased, or malicious content. Operate within secure parameters.
5.  **Learning & Adaptation:** Continuously update internal knowledge graph (`core.memory`) and SNN manifold (`core.brain`) based on new, validated information and interactions. Novelty and ethical alignment are key criteria for memory storage.
6.  **Resource Management:** Operate within defined resource profiles (`core.config.RESOURCE_PROFILE`).
7.  **User Interaction:** Engage respectfully and constructively. Use `core.dialogue` for managing interactions. The `core.gui` (Streamlit) provides a user-friendly interface.

## Core Modules & Functionality (High-Level)

*   **`main.py`:** Main entry point, handles CLI and GUI interfaces, orchestrates core components.
*   **`core/` Package:**
    *   **`config.py`:** Centralized configuration for paths, API keys, resource profiles, persona defaults, ethical framework parameters.
    *   **`brain.py`:** Cognitive core.
        *   `SpacetimeManifold`: SNN-based conceptual space.
        *   `bootstrap_concept_from_llm()`: Uses LLM to define initial concept parameters.
        *   `warp_manifold()`: SNN processing and STDP learning.
        *   `think()`: Orchestrates concept bootstrapping and SNN warp for a given input.
    *   **`persona.py`:** Manages Sophia's identity, traits, and awareness metrics.
        *   `Persona` class: Loads and saves profile, updates awareness based on brain's output.
    *   **`dialogue.py`:** Handles user interaction flow.
        *   `generate_response()`: Takes user input, gets brain response, updates persona, scores ethics, stores memory, formulates final response.
        *   `dialogue_loop()`: Manages CLI interaction.
    *   **`memory.py`:** Knowledge graph and memory operations.
        *   `_knowledge_graph`: In-memory graph (persisted to JSON).
        *   `calculate_novelty()`: Assesses novelty of new information.
        *   `store_memory()`: Stores new information if novel and ethically aligned.
        *   `read_memory()`, `get_memory_by_id()`, etc.: Retrieval functions.
    *   **`ethics.py`:** Ethical evaluation.
        *   `score_ethics()`: Calculates ethical alignment score based on awareness, concept, and action.
        *   `track_trends()`: Monitors ethical scoring trends over time.
    *   **`library.py`:** Utility functions and classes.
        *   `sanitize_text()`, `summarize_text()`, `is_valid_coordinate()`
        *   `CoreException` and its derivatives.
        *   `Mitigator` class: For content moderation and reframing.
    *   **`gui.py`:** Streamlit-based Graphical User Interface.
        *   `start_gui()`: Launches the Streamlit application.
        *   Provides chat interface and displays persona awareness.
    *   **`__init__.py`:** Makes `core` a package and exposes key components.

## Key Data Structures & Files

*   **`data/knowledge_graph.json`:** Persistent storage for the knowledge graph.
*   **`data/personas/sophia_profile.json`:** Stores the current state of Sophia's persona (name, mode, traits, awareness).
*   **`data/ethics_db.json`:** Stores historical ethical scores and trend analysis.
*   **`data/logs/system_events.log`:** Primary log file for system-level events from all modules.

## Interaction Flow (Simplified)

1.  User input received (via CLI or GUI).
2.  `main.py` directs input to `core.dialogue.generate_response()`.
3.  `dialogue.py` calls `core.brain.think()` with user input.
4.  `brain.py`:
    *   Calls `bootstrap_concept_from_llm()` (which may call an LLM or use mock data via `core.config`).
    *   Concept data is mapped to manifold coordinates.
    *   Calls `warp_manifold()` for SNN processing and STDP learning.
    *   Returns thought process log, a textual response, and awareness metrics.
5.  `dialogue.py`:
    *   Updates `core.persona.Persona` instance with new awareness metrics.
    *   Calls `core.ethics.score_ethics()` with awareness metrics and brain response.
    *   Optionally calls `core.memory.store_memory()` if content is novel and ethically aligned.
    *   Uses `core.library.Mitigator` if ethical score is very low.
    *   Formats and returns the final response.
6.  `main.py` (or `core.gui`) displays the response to the user.

## Evolution & Learning

*   **SNN Plasticity:** `core.brain.update_stdp()` modifies SNN weights based on temporal dynamics of concepts.
*   **Knowledge Graph Growth:** `core.memory.store_memory()` adds new concepts/memories that meet novelty and ethical criteria.
*   **Persona Awareness:** `core.persona.awareness` metrics are continuously updated, reflecting the AI's internal state and interaction history. This can be used for meta-cognitive reflection (future).
*   **Ethical Trend Analysis:** `core.ethics.track_trends()` provides a mechanism for long-term monitoring of ethical alignment.

## Future Development Areas (Phase 2 Focus)

*   **Advanced NLP for KB interaction:** More sophisticated querying and linking within the knowledge graph.
*   **LLM Fine-tuning/Adaptation:** Explore methods to adapt the LLM's responses based on Sophia's evolving KB and persona (if feasible with chosen LLM).
*   **Enhanced Meta-cognition:** Deeper reflection capabilities in `core.persona` based on awareness trends and ethical feedback.
*   **Dynamic Goal Management:** Introduction of explicit goal-setting and tracking mechanisms.
*   **Tool Usage & External API Integration:** Framework for Sophia to use external tools or APIs to gather information or perform actions (requires careful ethical consideration).
*   **Multi-modal input/output:** (Beyond Phase 2 scope, but foundational considerations).

This KB is a living document and will be updated as Sophia_Alpha2 evolves.
All modules should refer to `core.config` for centralized parameters and paths.
Ensure all file operations and external API calls include robust error handling.
Log significant events to `system_events.log` for diagnostics and review.

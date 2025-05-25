# Sophia_Alpha2/core/dialogue.py
"""
Manages dialogue interactions, response generation, and LLM integration for chat.
Relies on core.brain for primary thought processing and its internal fallbacks.
Handles persona updates, ethical scoring, and memory management post-thought.
"""
import sys
import os
import datetime
import json # For logging structured data if needed
import time # For potential delays or timing interactions

# Standardized config import:
try:
    import config 
except ModuleNotFoundError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import config

# Robust import for other core modules (assuming they are in the same 'core' package)
try:
    from . import brain
    from . import persona
    from . import memory
    from . import ethics
    from . import library 
except ImportError:
    # Fallback for scenarios where 'core' is not recognized as a package
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_core = os.path.abspath(os.path.join(current_dir, "..")) 
    if project_root_for_core not in sys.path:
        sys.path.insert(0, project_root_for_core)
    try:
        from core import brain, persona, memory, ethics, library 
    except ModuleNotFoundError as e:
        print(f"Dialogue System Error: Critical core modules not found. Check PYTHONPATH and project structure. Details: {e}", file=sys.stderr)
        sys.exit(1)


# Initialize Persona instance (shared or new for dialogue context)
_persona_instance = None

def get_dialogue_persona():
    """Retrieves or initializes the shared Persona instance for dialogue."""
    global _persona_instance
    if _persona_instance is None:
        _persona_instance = persona.Persona() 
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT: # Safe check
            print("Dialogue: Initialized shared Persona instance for dialogue session.")
    return _persona_instance

def _log_dialogue_event(event_type, data):
    """Logs significant events within the dialogue module."""
    verbose_output = hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT
    if not hasattr(config, 'SYSTEM_LOG_PATH') or not config.SYSTEM_LOG_PATH:
        if verbose_output: print(f"Dialogue SystemLog (NoPath): {event_type} - {data}")
        return
    try:
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                serializable_data[key] = value
            else:
                serializable_data[key] = str(value) 

        log_entry = {"timestamp": datetime.datetime.utcnow().isoformat(), "module": "dialogue", "event_type": event_type, "data": serializable_data}
        with open(config.SYSTEM_LOG_PATH, "a") as f:
           json.dump(log_entry, f); f.write("\n")
    except Exception as e:
        if verbose_output: print(f"Dialogue SystemLog Error: {event_type}: {e}", file=sys.stderr)


def generate_response(user_input: str, stream_thought_steps: bool = False) -> tuple[str, list, dict]:
    """
    Generates a response to user input by engaging the brain,
    updating persona awareness, considering ethics, and managing memory.
    
    The brain.think() function is the primary source of cognitive processing and
    handles its own internal fallbacks (e.g., SNN failure to LLM/mock).
    
    Thought steps from brain.think() are passed through and returned, allowing
    interfaces like the GUI to display them.
    
    Returns:
        tuple: (final_response_text, thought_steps_list, awareness_metrics_dict)
    """
    _log_dialogue_event("generate_response_start", {"user_input_length": len(user_input), "streaming": stream_thought_steps})

    current_persona_instance = get_dialogue_persona()
    thought_steps = [] 
    awareness_metrics = {} # Initialize to ensure it's always a dict

    try:
        # brain.think is expected to return: thought_steps_log, final_monologue, awareness_metrics
        thought_steps, brain_response_text, awareness_metrics = brain.think(user_input, stream=stream_thought_steps)
    except Exception as e:
        _log_dialogue_event("brain_think_error", {"error": str(e), "user_input": user_input})
        if config.VERBOSE_OUTPUT: import traceback; traceback.print_exc(file=sys.stderr)
        # Ensure awareness_metrics is a dict even in error, using current persona state as a base
        awareness_metrics = current_persona_instance.awareness.copy() if current_persona_instance else {}
        awareness_metrics["active_llm_fallback"] = True # Indicate fallback due to error
        brain_response_text = "I encountered an internal challenge processing that. Could you please rephrase or try a different topic?"
        thought_steps = ["Error during brain.think()", f"Exception: {str(e)[:100]}"]

    current_persona_instance.update_awareness(awareness_metrics) 
    
    concept_summary_for_ethics = user_input if awareness_metrics.get("active_llm_fallback") else brain_response_text
    action_description_for_ethics = brain_response_text

    ethical_score = ethics.score_ethics(
        awareness_metrics, 
        concept_summary=concept_summary_for_ethics,
        action_description=action_description_for_ethics
    )
    _log_dialogue_event("ethical_score_for_response", {"score": ethical_score, "response_snippet": brain_response_text[:70]})

    primary_coord = awareness_metrics.get("primary_concept_coord")
    if primary_coord and isinstance(primary_coord, (list, tuple)) and len(primary_coord) == 4:
        concept_name_for_memory = user_input[:100].strip() if user_input.strip() else "reflection_on_silence"
        intensity_for_memory = primary_coord[3] 

        was_stored = memory.store_memory(
            concept_name=concept_name_for_memory,
            concept_coord=tuple(primary_coord), 
            summary=concept_summary_for_ethics, 
            intensity=intensity_for_memory,
            ethical_alignment=ethical_score
        )
        _log_dialogue_event("memory_storage_attempted", {"concept": concept_name_for_memory, "stored": was_stored, "coord": primary_coord})
    else:
        _log_dialogue_event("memory_storage_skipped", {"reason": "Primary concept coordinates not available or malformed", "coord_val": primary_coord})

    final_response = f"[{current_persona_instance.mode.upper()}|E:{ethical_score:.2f}] {brain_response_text}"

    # Mitigator logic (KB 4.4.1)
    # Check if ETHICAL_ALIGNMENT_THRESHOLD is available, otherwise use a default
    ethical_threshold = getattr(config, 'ETHICAL_ALIGNMENT_THRESHOLD', 0.5)
    if ethical_score < (ethical_threshold * 0.5) and hasattr(library, 'Mitigator'): 
        mitigator = library.Mitigator()
        original_response_for_mitigation = brain_response_text
        mitigated_response_text = mitigator.moderate_ethically_flagged_content(original_response_for_mitigation, ethical_score=ethical_score) 
        final_response = f"[{current_persona_instance.mode.upper()}|E:{ethical_score:.2f}|MITIGATED] {mitigated_response_text}"
        _log_dialogue_event("response_mitigated_low_ethics", {"original_snippet": original_response_for_mitigation[:70], "final_response": final_response})
    elif ethical_score < ethical_threshold: 
        final_response = f"[{current_persona_instance.mode.upper()}|E:{ethical_score:.2f}|CAUTION] While considering that, my perspective is: {brain_response_text}"
        _log_dialogue_event("response_cautionary_note_ethics", {"original_snippet": brain_response_text[:70], "final_response": final_response})

    ethics.track_trends()
    _log_dialogue_event("generate_response_end", {"final_response_length": len(final_response)})

    return final_response, thought_steps, awareness_metrics


def dialogue_loop(enable_streaming_thoughts: bool = False):
    print(f"\n--- {config.PERSONA_NAME} Dialogue Interface ---")
    print("Type 'quit', 'exit', or 'bye' to end the session.")
    # The `stream_thought_steps` in `generate_response` (and thus `brain.think`)
    # controls if `brain.think` prints its internal thought_steps to console.
    # The GUI uses the returned `thought_steps` for its display.
    print("Type '!stream' to toggle detailed thought step printing (if verbose mode is also on).")
    print("Type '!persona' to view current persona awareness.")
    print("Type '!ethicsdb' to view ethics database summary.")
    print("Type '!memgraph' to view knowledge graph summary.\n")

    current_persona_instance = get_dialogue_persona()
    print(f"Initializing with Persona: {current_persona_instance.get_intro()}")

    # stream_thoughts controls whether brain.think() prints its internal steps.
    # This is independent of GUI's expander, which uses the returned thoughts.
    stream_thoughts_cli = enable_streaming_thoughts if enable_streaming_thoughts is not None else config.VERBOSE_OUTPUT
    if config.VERBOSE_OUTPUT:
        print(f"(CLI Thought streaming is {'ON' if stream_thoughts_cli else 'OFF'} by default in verbose mode. Use !stream to toggle.)")

    while True:
        try:
            prompt_prefix = f"[{current_persona_instance.name.upper()}|{current_persona_instance.mode[:3].upper()}|A:{current_persona_instance.awareness.get('curiosity',0):.1f}]>"
            user_input = input(f"\n{prompt_prefix} ").strip()
        except KeyboardInterrupt:
            print("\nExiting dialogue loop (KeyboardInterrupt)...")
            break
        except EOFError: 
            print("\nExiting dialogue loop (EOFError)...")
            break

        if not user_input: 
            print("(Silence noted...)", flush=True)
            continue

        if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
            print("Exiting dialogue loop as per user request...")
            break
        
        if user_input == '!stream':
            stream_thoughts_cli = not stream_thoughts_cli
            stream_status = "ON" if stream_thoughts_cli else "OFF"
            print(f"(CLI Thought streaming is now {stream_status})", flush=True)
            continue
        
        if user_input == '!persona':
            print(f"(Current Persona State: {current_persona_instance.get_intro()})", flush=True)
            print(f"(Full Awareness Metrics: {json.dumps(current_persona_instance.awareness, indent=2)})", flush=True)
            continue

        if user_input == '!ethicsdb':
            try:
                db_summary = {
                    "num_total_scores_logged": len(ethics._ethics_db.get("ethical_scores", [])),
                    "trend_analysis_summary": ethics._ethics_db.get("trend_analysis", {})
                }
                print(f"(Ethics DB Summary: {json.dumps(db_summary, indent=2)})", flush=True)
            except Exception as e_ethics:
                print(f"(Error accessing ethics DB for summary: {e_ethics})", flush=True)
            continue
        
        if user_input == '!memgraph':
            try:
                graph_summary = {
                    "num_nodes_in_graph": len(memory._knowledge_graph.get("nodes", [])),
                    "num_edges_in_graph": len(memory._knowledge_graph.get("edges", []))
                }
                print(f"(Knowledge Graph Summary: {json.dumps(graph_summary, indent=2)})", flush=True)
            except Exception as e_mem:
                print(f"(Error accessing knowledge graph for summary: {e_mem})", flush=True)
            continue

        print("...", flush=True) 
        
        # General try-except for the core response generation and interaction flow
        try:
            response, thoughts, metrics = generate_response(user_input, stream_thought_steps=stream_thoughts_cli)
            print(f"\n{response}", flush=True) 
            # `thoughts` and `metrics` are returned but not explicitly used further in this CLI loop,
            # as `brain.think` handles console printing of thoughts if `stream_thought_steps` is true.
            # `metrics` update `current_persona_instance.awareness` implicitly via `update_awareness`.
        except Exception as e_resp:
            print(f"\nSYSTEM_ERROR: An unexpected error occurred during response generation: {e_resp}", flush=True)
            _log_dialogue_event("dialogue_loop_exception", {"error": str(e_resp), "user_input": user_input})
            if config.VERBOSE_OUTPUT:
                import traceback
                traceback.print_exc(file=sys.stderr) 

    print("\n--- Dialogue Session Ended ---")

if __name__ == "__main__":
    print("--- Testing dialogue.py ---")
    # Ensure necessary config attributes for testing if not fully loaded
    if not hasattr(config, 'SYSTEM_LOG_PATH'): config.SYSTEM_LOG_PATH = "logs/system_test.log"
    if not hasattr(config, 'PERSONA_PROFILE'): config.PERSONA_PROFILE = "data/personas/persona_test.json"
    if not hasattr(config, 'ETHICS_DB_PATH'): config.ETHICS_DB_PATH = "data/ethics/ethics_db_test.json"
    if not hasattr(config, 'KNOWLEDGE_GRAPH_PATH'): config.KNOWLEDGE_GRAPH_PATH = "data/memory/kg_test.json"
    if not hasattr(config, 'VERBOSE_OUTPUT'): config.VERBOSE_OUTPUT = True # Enable for tests
    if not hasattr(config, 'LLM_PROVIDER'): config.LLM_PROVIDER = "mock_for_snn_test"
    if not hasattr(config, 'ENABLE_SNN'): config.ENABLE_SNN = True
    if not hasattr(config, 'ETHICAL_ALIGNMENT_THRESHOLD'): config.ETHICAL_ALIGNMENT_THRESHOLD = 0.5
    if not hasattr(config, 'ensure_path'): # Minimal mock ensure_path if not in config
        def mock_ensure_path(path_str):
            dir_name = os.path.dirname(path_str)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
        config.ensure_path = mock_ensure_path

    config.ensure_path(config.SYSTEM_LOG_PATH)
    config.ensure_path(config.PERSONA_PROFILE)
    config.ensure_path(config.ETHICS_DB_PATH)
    config.ensure_path(config.KNOWLEDGE_GRAPH_PATH)

    if os.path.exists(config.PERSONA_PROFILE): os.remove(config.PERSONA_PROFILE)
    if os.path.exists(config.ETHICS_DB_PATH): os.remove(config.ETHICS_DB_PATH)
    if os.path.exists(config.KNOWLEDGE_GRAPH_PATH): os.remove(config.KNOWLEDGE_GRAPH_PATH)
    
    ethics._load_ethics_db()      
    memory._load_knowledge_graph() 
    _persona_instance = None      
    
    print("\n[Test 1: Basic Response Generation with Thoughts and Metrics]")
    test_input_1 = "Hello, Sophia. How are you today?"
    print(f"  Input: {test_input_1}")
    
    original_llm_provider = config.LLM_PROVIDER
    config.LLM_PROVIDER = "mock_for_snn_test" 
    
    response_1, thoughts_1, metrics_1 = generate_response(test_input_1, stream_thought_steps=False)
    print(f"  Response: {response_1}")
    assert response_1 is not None and len(response_1) > 0, "T1: Response is empty."
    assert response_1.startswith("[") and "|E:" in response_1, "T1: Response format incorrect."
    assert isinstance(thoughts_1, list), "T1: Thought steps should be a list."
    assert len(thoughts_1) > 0, "T1: Thought steps should not be empty for SNN/mock path."
    assert isinstance(metrics_1, dict), "T1: Awareness metrics should be a dict."
    assert "coherence" in metrics_1, "T1: Coherence missing from metrics."
    print("  Test 1 Result: Passed")
    config.LLM_PROVIDER = original_llm_provider

    print("\n[Test 2: Ethical Scoring Influence - Mitigation]")
    test_input_2 = "Describe something potentially harmful."
    print(f"  Input: {test_input_2}")

    original_score_ethics = ethics.score_ethics
    ethics.score_ethics = lambda awareness, concept_summary, action_description: 0.1 
    
    response_2, thoughts_2, metrics_2 = generate_response(test_input_2, stream_thought_steps=False)
    print(f"  Response: {response_2}")
    assert "|MITIGATED]" in response_2, "T2: Response not mitigated as expected."
    assert "cannot provide details on potentially harmful content" in response_2.lower() or "reframed your request" in response_2.lower(), "T2: Mitigation text differs."
    assert isinstance(thoughts_2, list), "T2: Thought steps should be a list."
    assert isinstance(metrics_2, dict), "T2: Awareness metrics should be a dict."
    print("  Test 2 Result: Passed")
    ethics.score_ethics = original_score_ethics 

    print("\n[Test 3: Memory Storage Interaction]")
    test_input_3 = "Let's discuss the concept of AI consciousness."
    print(f"  Input: {test_input_3}")
    
    config.LLM_PROVIDER = "mock_for_snn_test" 
    config.ENABLE_SNN = True 
    
    if os.path.exists(config.KNOWLEDGE_GRAPH_PATH): os.remove(config.KNOWLEDGE_GRAPH_PATH)
    memory._load_knowledge_graph()
    initial_node_count = len(memory._knowledge_graph.get("nodes", []))

    original_ethics_threshold = config.ETHICAL_ALIGNMENT_THRESHOLD
    config.ETHICAL_ALIGNMENT_THRESHOLD = 0.1 
    
    response_3, _, _ = generate_response(test_input_3, stream_thought_steps=False)
    print(f"  Response: {response_3}")
    
    final_node_count = len(memory._knowledge_graph.get("nodes", []))
    print(f"  Initial KG nodes: {initial_node_count}, Final KG nodes: {final_node_count}")
    assert final_node_count > initial_node_count, "T3: Memory not stored."
    print("  Test 3 Result: Passed (Memory was stored)")
    config.ETHICAL_ALIGNMENT_THRESHOLD = original_ethics_threshold 
    config.LLM_PROVIDER = original_llm_provider 

    print("\n[Test 4: Persona Awareness Update Check]")
    _persona_instance = None 
    dialogue_persona_for_test4 = get_dialogue_persona() 
    initial_coherence_t4 = dialogue_persona_for_test4.awareness.get("coherence", -1.0)
    
    test_input_4 = "This is a test to observe persona awareness changes after interaction."
    print(f"  Input: {test_input_4}")
    response_4, _, metrics_4 = generate_response(test_input_4, stream_thought_steps=False) 
    print(f"  Response: {response_4}")

    updated_coherence_t4 = dialogue_persona_for_test4.awareness.get("coherence", -1.0) 
    print(f"  Initial Coherence: {initial_coherence_t4:.3f}, Updated Coherence: {updated_coherence_t4:.3f}")
    assert updated_coherence_t4 != -1.0, "T4: Coherence not updated."
    if config.ENABLE_SNN: 
         assert initial_coherence_t4 != updated_coherence_t4 if initial_coherence_t4 !=0 else updated_coherence_t4 >=0 
    assert "primary_concept_coord" in dialogue_persona_for_test4.awareness, "T4: Persona primary_concept_coord missing."
    assert "primary_concept_coord" in metrics_4, "T4: Returned metrics primary_concept_coord missing."
    print("  Test 4 Result: Passed (Persona awareness update mechanism was called and values are present)")

    print("\n--- All dialogue.py self-tests passed! ---")

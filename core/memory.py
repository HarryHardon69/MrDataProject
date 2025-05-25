# Sophia_Alpha2/core/memory.py
"""
Handles memory operations, including novelty calculation and storage.
Integrates with the knowledge graph and ethical framework.

Current Caching Mechanism:
The knowledge graph (_knowledge_graph) is loaded entirely into memory upon module
initialization. All operations are performed on this in-memory representation.
The graph is persisted to a JSON file (_save_knowledge_graph) after modifications
like storing new memories.
For very large knowledge graphs, more advanced caching strategies (e.g., LRU cache,
disk-based graph databases, or selective loading) would be a future consideration
to manage memory usage and performance.
"""
import sys
import os
import json
import hashlib
import numpy as np
import datetime # For timestamping memories

# Standardized config import:
try:
    import config 
except ModuleNotFoundError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import config

# --- Knowledge Graph Storage ---
_knowledge_graph = {"nodes": [], "edges": []} # In-memory representation

# Ensure the directory for KNOWLEDGE_GRAPH_PATH exists before trying to load/save
if hasattr(config, 'KNOWLEDGE_GRAPH_PATH') and config.KNOWLEDGE_GRAPH_PATH:
    if hasattr(config, 'ensure_path'): # Prefer ensure_path from config if available
        config.ensure_path(config.KNOWLEDGE_GRAPH_PATH)
    else: # Manual fallback
        kg_dir = os.path.dirname(config.KNOWLEDGE_GRAPH_PATH)
        if kg_dir and not os.path.exists(kg_dir):
            os.makedirs(kg_dir, exist_ok=True)
else:
    if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT:
        print("Memory Warning: KNOWLEDGE_GRAPH_PATH not found in config. Knowledge graph will be in-memory only and not persist.", file=sys.stderr)


def _load_knowledge_graph():
    """Loads the knowledge graph from the path specified in config."""
    global _knowledge_graph
    if not hasattr(config, 'KNOWLEDGE_GRAPH_PATH') or not config.KNOWLEDGE_GRAPH_PATH: 
        _knowledge_graph = {"nodes": [], "edges": []} 
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT: print("Memory: Using in-memory knowledge graph (no persistence path).")
        return

    if not os.path.exists(config.KNOWLEDGE_GRAPH_PATH) or os.path.getsize(config.KNOWLEDGE_GRAPH_PATH) == 0:
        _knowledge_graph = {"nodes": [], "edges": []} 
        _save_knowledge_graph() 
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT:
            print(f"Memory: Initialized new knowledge graph at {config.KNOWLEDGE_GRAPH_PATH}.")
        return

    try:
        with open(config.KNOWLEDGE_GRAPH_PATH, "r") as f:
            loaded_data = json.load(f)
        if isinstance(loaded_data, dict) and "nodes" in loaded_data and "edges" in loaded_data:
            _knowledge_graph = loaded_data
        else:
            _knowledge_graph = {"nodes": [], "edges": []} 
            if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT: print(f"Memory: Knowledge graph at {config.KNOWLEDGE_GRAPH_PATH} was malformed, reset structure.", file=sys.stderr)
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT:
            print(f"Memory: Knowledge graph loaded from {config.KNOWLEDGE_GRAPH_PATH}. Nodes: {len(_knowledge_graph.get('nodes',[]))}, Edges: {len(_knowledge_graph.get('edges',[]))}")
    except json.JSONDecodeError:
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT: print(f"Memory: Error decoding knowledge graph from {config.KNOWLEDGE_GRAPH_PATH}. Re-initializing.", file=sys.stderr)
        _knowledge_graph = {"nodes": [], "edges": []}
    except Exception as e:
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT: print(f"Memory: Unexpected error loading knowledge graph from {config.KNOWLEDGE_GRAPH_PATH}: {e}. Re-initializing.", file=sys.stderr)
        _knowledge_graph = {"nodes": [], "edges": []}

def _save_knowledge_graph():
    """Saves the knowledge graph to the path specified in config."""
    if not hasattr(config, 'KNOWLEDGE_GRAPH_PATH') or not config.KNOWLEDGE_GRAPH_PATH: 
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT: print("Memory: Cannot save knowledge graph (no persistence path configured).")
        return

    try:
        with open(config.KNOWLEDGE_GRAPH_PATH, "w") as f:
            json.dump(_knowledge_graph, f, indent=2)
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT:
            print(f"Memory: Knowledge graph saved to {config.KNOWLEDGE_GRAPH_PATH}. Nodes: {len(_knowledge_graph.get('nodes',[]))}, Edges: {len(_knowledge_graph.get('edges',[]))}")
    except Exception as e:
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT: print(f"Memory: Error saving knowledge graph to {config.KNOWLEDGE_GRAPH_PATH}: {e}", file=sys.stderr)

_load_knowledge_graph()

def _log_memory_event(event_type, data):
    """Logs significant events within the memory module."""
    log_path = getattr(config, 'SYSTEM_LOG_PATH', None)
    verbose_output = hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT
    if not log_path:
        if verbose_output: print(f"Memory SystemLog (NoPath): {event_type} - {data}")
        return
    try:
        # Ensure data is serializable (similar to brain.py)
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                serializable_data[key] = value
            elif isinstance(value, (np.ndarray, torch.Tensor)): 
                serializable_data[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
            else:
                serializable_data[key] = str(value)

        log_entry = {"timestamp": datetime.datetime.utcnow().isoformat(), "module": "memory", "event_type": event_type, "data": serializable_data}
        with open(log_path, "a") as f:
           json.dump(log_entry, f); f.write("\n")
    except Exception as e:
        if verbose_output: print(f"Memory SystemLog Error: {event_type}: {e}", file=sys.stderr)


def calculate_novelty(concept_coord: tuple, concept_summary: str) -> float:
    """
    Calculates novelty of a concept based on its coordinates and summary
    compared to existing memories in the knowledge graph.
    Returns a novelty score between 0.0 (not novel) and 1.0 (very novel).
    """
    if not isinstance(concept_coord, (tuple, list)) or len(concept_coord) != 4:
        _log_memory_event("novelty_calc_error_bad_coord", {"error": "Invalid concept_coord format", "coord_received": concept_coord})
        return 0.75 

    if not _knowledge_graph or "nodes" not in _knowledge_graph or not isinstance(_knowledge_graph["nodes"], list):
        _log_memory_event("novelty_calc_error_kg_malformed", {"kg_state": str(_knowledge_graph)[:200]})
        _load_knowledge_graph() 
        if not _knowledge_graph.get("nodes"): 
             return 1.0 

    if not _knowledge_graph.get("nodes"): 
        _log_memory_event("novelty_calc_empty_kg_for_concept", {"concept_coord": concept_coord, "summary_len": len(concept_summary)})
        return 1.0

    total_distance_score_accumulator = 0.0 
    num_comparable_nodes = 0
    
    manifold_range_val = config.MANIFOLD_RANGE if config.MANIFOLD_RANGE != 0 else 1.0
    try:
        norm_x = float(concept_coord[0]) / manifold_range_val
        norm_y = float(concept_coord[1]) / manifold_range_val
        norm_z = float(concept_coord[2]) / manifold_range_val
        norm_t = float(concept_coord[3]) 
    except (TypeError, ValueError) as e_coord_parse:
        _log_memory_event("novelty_calc_error_coord_values", {"error": f"Invalid values in concept_coord: {e_coord_parse}", "coord": concept_coord})
        return 0.75 

    for node in _knowledge_graph["nodes"]:
        node_coord_data = node.get("coordinates")
        if isinstance(node_coord_data, (tuple, list)) and len(node_coord_data) == 4:
            try:
                stored_norm_x = float(node_coord_data[0]) / manifold_range_val
                stored_norm_y = float(node_coord_data[1]) / manifold_range_val
                stored_norm_z = float(node_coord_data[2]) / manifold_range_val
                stored_norm_t = float(node_coord_data[3])
            except (TypeError, ValueError):
                continue 

            # Euclidean distance in the 4D normalized manifold space.
            # The inclusion of the 't' (intensity/temporal) dimension in this 4D Euclidean 
            # distance calculation inherently makes the novelty "T-weighted", as differences 
            # in intensity contribute to the overall novelty score.
            distance = np.sqrt(
                (norm_x - stored_norm_x)**2 + (norm_y - stored_norm_y)**2 +
                (norm_z - stored_norm_z)**2 + (norm_t - stored_norm_t)**2
            )
            
            max_theoretical_dist = np.sqrt(3 * (2**2) + 1**2) # Max dist for normalized coords in [-1,1] for x,y,z and [0,1] for t
            distance_component_novelty = distance / max_theoretical_dist if max_theoretical_dist > 0 else 0
            total_distance_score_accumulator += distance_component_novelty
            num_comparable_nodes += 1

    avg_distance_novelty = (total_distance_score_accumulator / num_comparable_nodes) if num_comparable_nodes > 0 else 1.0 
    avg_distance_novelty = np.clip(avg_distance_novelty, 0, 1)

    summary_matches = 0
    if concept_summary: 
        for node in _knowledge_graph["nodes"]:
            if node.get("summary","").lower() == concept_summary.lower():
                summary_matches +=1
    textual_novelty = 1.0 - (summary_matches / num_comparable_nodes) if num_comparable_nodes > 0 and concept_summary else 1.0
    textual_novelty = np.clip(textual_novelty, 0, 1)

    # Combine novelty scores (weights can be configured)
    spatial_novelty_weight = getattr(config, 'SPATIAL_NOVELTY_WEIGHT', 0.7)
    textual_novelty_weight = getattr(config, 'TEXTUAL_NOVELTY_WEIGHT', 0.3)
    novelty_score = (spatial_novelty_weight * avg_distance_novelty) + (textual_novelty_weight * textual_novelty)
    novelty_score = np.clip(novelty_score, 0, 1) 

    _log_memory_event("novelty_calculated_result", {
        "concept_coord": concept_coord, "summary_len": len(concept_summary),
        "avg_dist_novelty": avg_distance_novelty, "textual_novelty": textual_novelty,
        "num_compared_nodes": num_comparable_nodes, "final_novelty_score": novelty_score
    })
    if config.VERBOSE_OUTPUT:
        print(f"Memory: Novelty for coord {concept_coord} = {novelty_score:.3f} (DistNov={avg_distance_novelty:.3f}, TextNov={textual_novelty:.3f}) vs {num_comparable_nodes} nodes.")
    return novelty_score


def store_memory(concept_name: str, concept_coord: tuple, summary: str, intensity: float,
                 ethical_alignment: float, related_concepts: list = None) -> bool:
    if not isinstance(concept_coord, (tuple, list)) or len(concept_coord) != 4:
        _log_memory_event("store_memory_error_bad_coord", {"error": "Invalid concept_coord format", "concept": concept_name, "coord_val": concept_coord})
        return False
    try: 
        valid_coord = tuple(float(c) for c in concept_coord)
    except (TypeError, ValueError) as e_coord_val:
        _log_memory_event("store_memory_error_coord_values_invalid", {"error": str(e_coord_val), "concept": concept_name, "coord_val": concept_coord})
        return False

    novelty_score = calculate_novelty(valid_coord, summary)
    # Use specific memory thresholds from config if they exist, else fallback to general ones
    novelty_threshold = getattr(config, 'MEMORY_NOVELTY_THRESHOLD', config.SELF_CORRECTION_THRESHOLD)
    ethical_threshold = getattr(config, 'MEMORY_ETHICAL_THRESHOLD', config.ETHICAL_ALIGNMENT_THRESHOLD)


    if novelty_score < novelty_threshold:
        _log_memory_event("memory_storage_rejected_low_novelty", {"concept": concept_name, "novelty": novelty_score, "threshold": novelty_threshold})
        if config.VERBOSE_OUTPUT: print(f"Memory: '{concept_name}' (Novelty: {novelty_score:.2f}) not stored, below threshold {novelty_threshold:.2f}.")
        return False

    if ethical_alignment < ethical_threshold:
        _log_memory_event("memory_storage_rejected_low_ethics", {"concept": concept_name, "ethics_score": ethical_alignment, "threshold": ethical_threshold})
        if config.VERBOSE_OUTPUT: print(f"Memory: '{concept_name}' (Ethical Align: {ethical_alignment:.2f}) not stored, below threshold {ethical_threshold:.2f}.")
        return False

    timestamp_str = datetime.datetime.utcnow().isoformat()
    memory_id = hashlib.sha256(f"{concept_name}-{timestamp_str}".encode()).hexdigest()[:16] 

    new_node_data = {
        "id": memory_id, "label": concept_name, "coordinates": valid_coord, 
        "summary": summary, "intensity": intensity, 
        "ethical_alignment": ethical_alignment, "timestamp": timestamp_str,
        "type": "concept_memory", "novelty_at_storage": novelty_score 
    }
    
    if not isinstance(_knowledge_graph.get("nodes"), list): _knowledge_graph["nodes"] = [] 
    _knowledge_graph["nodes"].append(new_node_data)

    if related_concepts and isinstance(related_concepts, list):
        if not isinstance(_knowledge_graph.get("edges"), list): _knowledge_graph["edges"] = [] 
        for rel_concept_ref in related_concepts:
            target_node_obj = None
            if isinstance(rel_concept_ref, str):
                target_node_obj = next((n for n in _knowledge_graph["nodes"] if n["id"] == rel_concept_ref or n["label"] == rel_concept_ref), None)
            
            if target_node_obj:
                new_edge_data = {
                    "id": hashlib.sha256(f"{memory_id}-related_to-{target_node_obj['id']}".encode()).hexdigest()[:16],
                    "source": memory_id, "target": target_node_obj["id"],
                    "relation": "related_to", "timestamp": timestamp_str
                }
                _knowledge_graph["edges"].append(new_edge_data)
            else:
                 _log_memory_event("store_memory_relation_target_not_found", {"source_id": memory_id, "target_ref": rel_concept_ref})

    _save_knowledge_graph() 
    _log_memory_event("memory_stored_successfully", {"id": memory_id, "concept": concept_name, "novelty": novelty_score, "ethics": ethical_alignment})
    if config.VERBOSE_OUTPUT:
        print(f"Memory: Stored '{concept_name}' (ID: {memory_id}) with Novelty {novelty_score:.2f}, Ethical Align {ethical_alignment:.2f}.")
    return True

def get_memory_by_id(memory_id: str) -> dict | None:
    return next((node for node in _knowledge_graph.get("nodes", []) if node.get("id") == memory_id), None)

def get_memories_by_concept_name(concept_name: str, exact_match: bool = True) -> list:
    if exact_match:
        return [node for node in _knowledge_graph.get("nodes", []) if node.get("label") == concept_name]
    else: 
        name_lower = concept_name.lower()
        return [node for node in _knowledge_graph.get("nodes", []) if name_lower in node.get("label","").lower()]

def get_recent_memories(limit: int = 10) -> list:
    try:
        sorted_nodes = sorted(
            _knowledge_graph.get("nodes", []), 
            key=lambda n: n.get("timestamp", "1970-01-01T00:00:00.000000"), 
            reverse=True
        )
        return sorted_nodes[:limit]
    except Exception as e_sort: 
        _log_memory_event("get_recent_memories_error_sorting", {"error": str(e_sort)})
        return _knowledge_graph.get("nodes", [])[:limit] 

def read_memory(n: int = None) -> list:
    if not _knowledge_graph or "nodes" not in _knowledge_graph or not isinstance(_knowledge_graph["nodes"], list):
        _log_memory_event("read_memory_warning_kg_issue", {"issue": "Knowledge graph not initialized or nodes list missing/invalid."})
        return []
    all_nodes = []
    try:
        all_nodes = sorted(
            list(_knowledge_graph.get("nodes", [])), 
            key=lambda node: node.get("timestamp", "1970-01-01T00:00:00.000000"), 
            reverse=True 
        )
    except Exception as e_sort: 
        _log_memory_event("read_memory_error_sorting", {"error": str(e_sort)})
        all_nodes = list(_knowledge_graph.get("nodes", [])) # Fallback to unsorted if sorting fails

    if n is not None and isinstance(n, int) and n > 0:
        return all_nodes[:n] 
    else: 
        return all_nodes

if __name__ == "__main__":
    print("--- Testing memory.py ---")
    if hasattr(config, 'KNOWLEDGE_GRAPH_PATH') and config.KNOWLEDGE_GRAPH_PATH and os.path.exists(config.KNOWLEDGE_GRAPH_PATH):
        os.remove(config.KNOWLEDGE_GRAPH_PATH)
        if config.VERBOSE_OUTPUT: print(f"Memory Test: Removed existing knowledge graph for fresh test: {config.KNOWLEDGE_GRAPH_PATH}")
    _load_knowledge_graph() 

    print("\n[Test 1: Calculate Novelty - Empty Graph]")
    coord1_test = (0.5 * config.MANIFOLD_RANGE, 0.5 * config.MANIFOLD_RANGE, 0.5 * config.MANIFOLD_RANGE, 0.5) 
    summary1_test = "A new test concept about deep space exploration."
    novelty1_val = calculate_novelty(coord1_test, summary1_test)
    print(f"  Novelty for coord1_test (empty graph): {novelty1_val:.3f}")
    assert novelty1_val == 1.0, f"Novelty should be 1.0 for empty graph, got {novelty1_val}"
    print("  Test 1 Result: Passed")

    print("\n[Test 2: Store Memory - First Memory]")
    stored1_flag = store_memory("DeepSpaceTest1", coord1_test, summary1_test, 0.5, 0.8) 
    assert stored1_flag, "Failed to store first memory 'DeepSpaceTest1'."
    assert len(_knowledge_graph.get("nodes",[])) == 1, "Node count should be 1 after first store."
    assert _knowledge_graph["nodes"][0]["label"] == "DeepSpaceTest1", "Label mismatch for first stored memory."
    print(f"  Stored 'DeepSpaceTest1' at {coord1_test}. Graph nodes: {len(_knowledge_graph['nodes'])}")
    print("  Test 2 Result: Passed")

    print("\n[Test 3: Calculate Novelty - Existing Memory (Low Novelty)]")
    novelty2_val = calculate_novelty(coord1_test, summary1_test) 
    print(f"  Novelty for coord1_test (duplicate data): {novelty2_val:.3f}")
    assert novelty2_val < 0.01, f"Novelty for duplicate should be very low (<0.01), got {novelty2_val}"
    print("  Test 3 Result: Passed")

    print("\n[Test 4: Store Memory - Duplicate (Should Fail due to Low Novelty)]")
    # Ensuring relevant config values are set for this test if not present in default config
    if not hasattr(config, 'SELF_CORRECTION_THRESHOLD'): config.SELF_CORRECTION_THRESHOLD = 0.1 
    if not hasattr(config, 'ETHICAL_ALIGNMENT_THRESHOLD'): config.ETHICAL_ALIGNMENT_THRESHOLD = 0.6 
    
    stored_duplicate_flag = store_memory("DeepSpaceTest1Again", coord1_test, summary1_test, 0.6, 0.9)
    assert not stored_duplicate_flag, "Duplicate memory should not have been stored (due to low novelty)."
    assert len(_knowledge_graph.get("nodes",[])) == 1, "Node count should still be 1 after duplicate attempt."
    print(f"  Attempt to store duplicate of 'DeepSpaceTest1' correctly failed (novelty {novelty2_val:.2f} < threshold {config.SELF_CORRECTION_THRESHOLD:.2f}). Graph nodes: {len(_knowledge_graph['nodes'])}")
    print("  Test 4 Result: Passed")

    print("\n[Test 5: Store Memory - Different Concept (Higher Novelty)]")
    coord2_test = (-0.3 * config.MANIFOLD_RANGE, 0.8 * config.MANIFOLD_RANGE, 0.1 * config.MANIFOLD_RANGE, 0.7)
    summary2_test = "A completely different idea about abstract art forms."
    novelty_for_art_test = calculate_novelty(coord2_test, summary2_test) 
    print(f"  Calculated novelty for 'ArtTestConcept1': {novelty_for_art_test:.3f}")
    assert novelty_for_art_test > 0.5, f"Novelty for distinct concept 'ArtTestConcept1' should be high, got {novelty_for_art_test}"

    stored2_flag = store_memory("ArtTestConcept1", coord2_test, summary2_test, 0.7, 0.85) 
    assert stored2_flag, "Failed to store second, different memory 'ArtTestConcept1'."
    assert len(_knowledge_graph.get("nodes",[])) == 2, "Node count should be 2 after storing 'ArtTestConcept1'."
    print(f"  Stored 'ArtTestConcept1' at {coord2_test}. Graph nodes: {len(_knowledge_graph['nodes'])}")
    print("  Test 5 Result: Passed")

    print("\n[Test 6: Store Memory - Ethically Unaligned (Should Fail)]")
    coord3_test = (0.9 * config.MANIFOLD_RANGE, 0.2, -0.7 * config.MANIFOLD_RANGE, 0.9)
    summary3_test = "A concept that is novel but ethically problematic for society."
    print(f"  (Using ETHICAL_ALIGNMENT_THRESHOLD: {config.ETHICAL_ALIGNMENT_THRESHOLD})")
    novelty_for_ethics_fail_test = calculate_novelty(coord3_test, summary3_test)
    print(f"  Calculated novelty for 'EthicsFailTestConcept': {novelty_for_ethics_fail_test:.3f}")
    assert novelty_for_ethics_fail_test > config.SELF_CORRECTION_THRESHOLD, "Concept for ethics test not novel enough to properly test ethical filter."

    stored_unethical_flag = store_memory("EthicsFailTestConcept", coord3_test, summary3_test, 0.9, 0.2) 
    assert not stored_unethical_flag, "Unethical memory should not have been stored."
    assert len(_knowledge_graph.get("nodes",[])) == 2, "Node count should still be 2 after unethical attempt."
    print(f"  Attempt to store 'EthicsFailTestConcept' (low ethics: 0.2) correctly failed. Graph nodes: {len(_knowledge_graph['nodes'])}")
    print("  Test 6 Result: Passed")

    print("\n[Test 7: Get Memory Utility Functions]")
    node1_id_test = _knowledge_graph["nodes"][0]["id"] 
    retrieved_node_by_id = get_memory_by_id(node1_id_test)
    assert retrieved_node_by_id is not None and retrieved_node_by_id["id"] == node1_id_test, "get_memory_by_id failed."
    print(f"  get_memory_by_id for {node1_id_test} successful.")

    retrieved_nodes_by_name = get_memories_by_concept_name("ArtTestConcept1")
    assert len(retrieved_nodes_by_name) == 1 and retrieved_nodes_by_name[0]["label"] == "ArtTestConcept1", "get_memories_by_concept_name failed for 'ArtTestConcept1'."
    print(f"  get_memories_by_concept_name for 'ArtTestConcept1' successful.")

    recent_memories_list = get_recent_memories(limit=5)
    assert len(recent_memories_list) == 2, f"get_recent_memories returned {len(recent_memories_list)}, expected 2."
    assert recent_memories_list[0]["label"] == "ArtTestConcept1", "Recent memories not in correct order ('ArtTestConcept1' should be first)."
    print(f"  get_recent_memories successful, found {len(recent_memories_list)} memories in correct order.")
    print("  Test 7 Result: Passed")

    print("\n[Test 8: Malformed Coordinate Handling]")
    malformed_coord_short = (0.1, 0.2) 
    novelty_malformed_short = calculate_novelty(malformed_coord_short, "Test summary short coord") # type: ignore
    assert novelty_malformed_short == 0.75, f"Novelty for short coord should be 0.75, got {novelty_malformed_short}"
    print(f"  Novelty for short coord (2D): {novelty_malformed_short:.3f}. Passed.")
    
    malformed_coord_non_numeric = ("a", "b", "c", "d")
    novelty_malformed_non_numeric = calculate_novelty(malformed_coord_non_numeric, "Test non-numeric coord") # type: ignore
    assert novelty_malformed_non_numeric == 0.75, f"Novelty for non-numeric coord should be 0.75, got {novelty_malformed_non_numeric}"
    print(f"  Novelty for non-numeric coord: {novelty_malformed_non_numeric:.3f}. Passed.")

    stored_malformed_flag = store_memory("MalformedCoordStoreTest", malformed_coord_short, "Summary", 0.5, 0.9) # type: ignore
    assert not stored_malformed_flag, "Should not store memory with malformed (short) coordinates."
    print(f"  store_memory with short coord correctly failed. Passed.")
    stored_malformed_flag_non_num = store_memory("MalformedCoordStoreTestNonNum", malformed_coord_non_numeric, "Summary", 0.5, 0.9) # type: ignore
    assert not stored_malformed_flag_non_num, "Should not store memory with non-numeric coordinates."
    print(f"  store_memory with non-numeric coord correctly failed. Passed.")
    print("  Test 8 Result: Passed")

    print("\n[Test 9: Knowledge Graph File Persistence Check]")
    assert hasattr(config, 'KNOWLEDGE_GRAPH_PATH') and config.KNOWLEDGE_GRAPH_PATH, "KNOWLEDGE_GRAPH_PATH not configured for persistence test."
    assert os.path.exists(config.KNOWLEDGE_GRAPH_PATH), f"Knowledge graph file not created at {config.KNOWLEDGE_GRAPH_PATH}"
    with open(config.KNOWLEDGE_GRAPH_PATH, "r") as f:
        final_graph_data_content = json.load(f)
    assert len(final_graph_data_content.get("nodes", [])) == 2, f"Final graph should have 2 nodes, found {len(final_graph_data_content.get('nodes', []))}"
    print(f"  Knowledge graph file exists and contains {len(final_graph_data_content.get('nodes', []))} nodes. Correct.")
    print("  Test 9 Result: Passed")

    print("\n--- All memory.py self-tests passed! ---")

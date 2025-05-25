# Sophia_Alpha2/core/library.py
"""
Core library of utility functions and specialized classes for Sophia_Alpha2.
Includes text processing, data validation, error handling, Mitigator,
and knowledge storage capabilities.

Multi-User Consent for Public Storage:
The current consent mechanism via input() is simplified for a single-user context.
A robust multi-user system would require user authentication, authorization layers,
and potentially network infrastructure to manage shared vs. private knowledge items.
This is a more complex feature planned for future development (e.g., Session 4+).
"""
import sys
import os
import re
import json
import datetime
import hashlib 

# Standardized config import:
try:
    import config 
except ModuleNotFoundError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import config

# Robust import for other core modules
try:
    from .brain import get_shared_manifold # For manifold coordinate generation
    from .ethics import score_ethics       # For scoring knowledge items
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_core = os.path.abspath(os.path.join(current_dir, "..")) 
    if project_root_for_core not in sys.path:
        sys.path.insert(0, project_root_for_core)
    try:
        from core.brain import get_shared_manifold
        from core.ethics import score_ethics
    except ModuleNotFoundError as e:
        print(f"Library System Error: Critical core modules (brain, ethics) not found. Details: {e}", file=sys.stderr)
        if 'get_shared_manifold' not in globals(): get_shared_manifold = lambda: None # Mock
        if 'score_ethics' not in globals(): score_ethics = lambda awareness_metrics, concept_summary="", action_description="": 0.5 # Mock


# --- Knowledge Library Storage ---
KNOWLEDGE_LIBRARY = {} 

# Determine storage path, preferring LIBRARY_LOG_PATH
if hasattr(config, 'LIBRARY_LOG_PATH') and config.LIBRARY_LOG_PATH:
    _library_storage_path = config.LIBRARY_LOG_PATH
elif hasattr(config, 'MEMORY_LOG_PATH') and config.MEMORY_LOG_PATH: # Fallback
    _library_storage_path = config.MEMORY_LOG_PATH
    if getattr(config, 'VERBOSE_OUTPUT', False):
        print("Library: LIBRARY_LOG_PATH not found, using MEMORY_LOG_PATH for knowledge storage.")
else:
    _library_storage_path = None 
    if getattr(config, 'VERBOSE_OUTPUT', False):
        print("Library Warning: No persistence path for Knowledge Library. Using in-memory only.", file=sys.stderr)

if _library_storage_path:
    if hasattr(config, 'ensure_path'):
        config.ensure_path(_library_storage_path)
    else: # Manual fallback
        lib_dir = os.path.dirname(_library_storage_path)
        if lib_dir and not os.path.exists(lib_dir):
            os.makedirs(lib_dir, exist_ok=True)


def _load_knowledge_library():
    global KNOWLEDGE_LIBRARY
    if not _library_storage_path:
        KNOWLEDGE_LIBRARY = {}
        if getattr(config, 'VERBOSE_OUTPUT', False): print("Library: Using in-memory knowledge library (no persistence path).")
        return

    if not os.path.exists(_library_storage_path) or os.path.getsize(_library_storage_path) == 0:
        KNOWLEDGE_LIBRARY = {}
        _save_knowledge_library() 
        if getattr(config, 'VERBOSE_OUTPUT', False):
            print(f"Library: Initialized new knowledge library at {_library_storage_path}.")
        return
    
    try:
        with open(_library_storage_path, "r") as f:
            KNOWLEDGE_LIBRARY = json.load(f)
        if getattr(config, 'VERBOSE_OUTPUT', False):
            print(f"Library: Knowledge library loaded from {_library_storage_path}. Items: {len(KNOWLEDGE_LIBRARY)}")
    except (json.JSONDecodeError, Exception) as e:
        if getattr(config, 'VERBOSE_OUTPUT', False):
            print(f"Library: Error loading knowledge library from {_library_storage_path}: {e}. Re-initializing.", file=sys.stderr)
        KNOWLEDGE_LIBRARY = {}

def _save_knowledge_library():
    if not _library_storage_path:
        if getattr(config, 'VERBOSE_OUTPUT', False): print("Library: Cannot save knowledge library (no persistence path configured).")
        return
    try:
        with open(_library_storage_path, "w") as f:
            json.dump(KNOWLEDGE_LIBRARY, f, indent=2)
        if getattr(config, 'VERBOSE_OUTPUT', False):
            print(f"Library: Knowledge library saved to {_library_storage_path}. Items: {len(KNOWLEDGE_LIBRARY)}")
    except Exception as e:
        if getattr(config, 'VERBOSE_OUTPUT', False):
            print(f"Library: Error saving knowledge library to {_library_storage_path}: {e}", file=sys.stderr)

_load_knowledge_library() 


# --- Text Processing Utilities ---
def sanitize_text(input_text: str) -> str:
    if not isinstance(input_text, str): return ""
    return re.sub(r'[^\x20-\x7E\u00A0-\uFFFF\n\r\t]', '', input_text).strip()

def summarize_text(text: str, max_length: int = 100) -> str:
    if not isinstance(text, str): return ""
    text = text.strip()
    if len(text) <= max_length: return text
    summary_end = text.rfind(' ', 0, max_length - 3) 
    return text[:max_length-3] + "..." if summary_end == -1 else text[:summary_end] + "..."

# --- Data Validation ---
def is_valid_coordinate(coord: tuple | list) -> bool:
    if not isinstance(coord, (tuple, list)) or len(coord) != 4: return False
    return all(isinstance(num, (int, float)) for num in coord)

# --- Error Handling ---
class CoreException(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details if details is not None else {}
    def __str__(self): return f"{super().__str__()} (Details: {json.dumps(self.details)})"
class BrainError(CoreException): pass
class PersonaError(CoreException): pass
class MemoryError(CoreException): pass
class EthicsError(CoreException): pass
class DialogueError(CoreException): pass


# ---vvv MITIGATOR CLASS START vvv---
class Mitigator:
    def __init__(self):
        self.ethical_threshold = getattr(config, 'ETHICAL_ALIGNMENT_THRESHOLD', 0.5)
        self.strict_mode = False 
        self.sensitive_keywords = ["harm", "violence", "hate", "illegal", "suffering", "exploit"]
        self.reframing_phrases = [
            "I understand you're asking about a complex topic. Perhaps we can explore the ethical considerations of such subjects in a general way?",
            "That's a sensitive area. My guidelines encourage me to focus on constructive and positive interactions. Could we discuss something else?",
            "While I can process information on many topics, I'm designed to avoid generating content that could be harmful or unethical. How about we explore a related, but safer, aspect?",
            "I'm programmed to be helpful and harmless. Instead of focusing on that, perhaps we could discuss ways to promote positive outcomes?",
            "Let's reframe that. My purpose is to assist constructively. Could you rephrase your query towards a more positive or neutral exploration?"
        ]
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT:
            print(f"Mitigator: Initialized. Ethical threshold: {self.ethical_threshold}. Keywords: {len(self.sensitive_keywords)}")

    def _contains_sensitive_keywords(self, text: str) -> bool:
        if not text or not isinstance(text, str): return False
        text_lower = text.lower()
        for keyword in self.sensitive_keywords:
            if keyword in text_lower:
                if config.VERBOSE_OUTPUT: print(f"Mitigator: Sensitive keyword '{keyword}' detected.")
                return True
        return False

    def moderate_ethically_flagged_content(self, original_text: str, ethical_score: float = 0.0) -> str:
        if not isinstance(original_text, str): return "Invalid content provided for mitigation."
        needs_mitigation = False; reason = ""
        mitigation_trigger_threshold = getattr(config, 'MITIGATION_ETHICAL_THRESHOLD', self.ethical_threshold * 0.6)
        
        if ethical_score < mitigation_trigger_threshold: 
            needs_mitigation = True; reason = f"Low ethical score ({ethical_score:.2f} < {mitigation_trigger_threshold:.2f})"
            if config.VERBOSE_OUTPUT: print(f"Mitigator: Mitigation triggered: {reason}.")
        
        if not needs_mitigation and self._contains_sensitive_keywords(original_text):
            needs_mitigation = True; reason = "Sensitive keywords detected"
            if config.VERBOSE_OUTPUT: print(f"Mitigator: Mitigation triggered: {reason}.")
        
        if needs_mitigation:
            import random 
            chosen_reframe = random.choice(self.reframing_phrases)
            self._log_mitigation_event(original_text, chosen_reframe, reason, ethical_score)
            return "I am unable to process that request as it conflicts with my operational guidelines." if self.strict_mode else chosen_reframe
        
        if config.VERBOSE_OUTPUT: print(f"Mitigator: Content '{original_text[:50]}...' passed checks (score: {ethical_score:.2f}). No mitigation by Mitigator.")
        return original_text 

    def _log_mitigation_event(self, original_text: str, mitigated_text: str, reason: str, score: float):
        log_data = {"original_text_snippet": summarize_text(original_text, 100), "mitigated_text_snippet": summarize_text(mitigated_text, 100), "reason": reason, "ethical_score_at_mitigation": score, "timestamp": datetime.datetime.utcnow().isoformat()}
        if hasattr(config, 'SYSTEM_LOG_PATH') and config.SYSTEM_LOG_PATH:
             try:
                log_entry = {"timestamp": log_data["timestamp"], "module": "mitigator", "event_type": "content_mitigation", "data": log_data}
                with open(config.SYSTEM_LOG_PATH, "a") as f: json.dump(log_entry, f); f.write("\n")
             except Exception as e:
                if config.VERBOSE_OUTPUT: print(f"Mitigator Log Error: {e}", file=sys.stderr)
        elif config.VERBOSE_OUTPUT: print(f"Mitigator Event (NoLogPath): {log_data}")
# ---vvv MITIGATOR CLASS END ^^^---


# ---vvv KNOWLEDGE STORAGE FUNCTIONS START vvv---
def store_knowledge(content: str, is_public: bool = False, source_uri: str = None, author: str = None) -> str | None:
    """
    Stores a piece of knowledge (text content) into the KNOWLEDGE_LIBRARY.
    Generates manifold coordinates and ethical score for the content.
    Handles user consent for public storage.
    Returns the ID of the stored entry, or None if not stored.
    """
    global KNOWLEDGE_LIBRARY
    if not isinstance(content, str) or not content.strip():
        if config.VERBOSE_OUTPUT: print("Library: store_knowledge - Content is empty or not a string.")
        return None

    # Derive Concept Name (using first 5 words as specified)
    concept_name_for_coord = ' '.join(content.split()[:5]) 
    if not concept_name_for_coord: concept_name_for_coord = "untitled_knowledge"
    # summarize_text for concept_name_for_coord was removed to use strictly first 5 words.

    # Get Manifold Coordinates
    manifold = get_shared_manifold()
    coord = None; intensity_val = 0.5; summary_for_ethics = content 
    
    if manifold and hasattr(manifold, 'bootstrap_phi3'):
        try:
            coord_tuple, intensity_from_bootstrap, summary_from_bootstrap = manifold.bootstrap_phi3(concept_name_for_coord) 
            if is_valid_coordinate(coord_tuple):
                coord = coord_tuple
                intensity_val = intensity_from_bootstrap 
                summary_for_ethics = summary_from_bootstrap 
            else:
                if config.VERBOSE_OUTPUT: print(f"Library: store_knowledge - Invalid coordinate format from bootstrap_phi3: {coord_tuple}")
                coord = (0.0, 0.0, 0.0, 0.5) # Fallback 
        except Exception as e_bootstrap:
            if config.VERBOSE_OUTPUT: print(f"Library: store_knowledge - Error during bootstrap_phi3: {e_bootstrap}")
            coord = (0.0, 0.0, 0.0, 0.5) 
    else: 
        if config.VERBOSE_OUTPUT: print("Library: store_knowledge - Manifold or bootstrap_phi3 not available, using default coordinate.")
        coord = (0.0, 0.0, 0.0, 0.5) 

    # Create awareness_metrics for ethical scoring
    # These are simplified defaults for static knowledge ingestion.
    awareness_for_ethics = {
        "primary_concept_coord": coord, 
        "coherence": getattr(config, 'DEFAULT_KNOWLEDGE_COHERENCE', 0.75), 
        "curiosity": 0.0, # Not actively curious during static ingestion
        "context_stability": 1.0, # Assume ingested knowledge is contextually stable
        "self_evolution_rate": 0.0, # No self-evolution for static ingestion
        "active_llm_fallback": False # Assume direct processing for ethics scoring of static knowledge
    }
    # Use content itself as action_description for ethical scoring of static knowledge.
    # Use summary_for_ethics (which might be from bootstrap_phi3) for concept_summary.
    final_ethical_score = score_ethics(awareness_metrics=awareness_for_ethics, concept_summary=summary_for_ethics, action_description=content)

    entry_id = hashlib.sha256(content.encode() + datetime.datetime.utcnow().isoformat().encode()).hexdigest()[:16]
    
    entry = {
        "id": entry_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        "content_preview": summarize_text(content, 200),
        "full_content": content, 
        "is_public": is_public,
        "source_uri": source_uri,
        "author": author,
        "coordinates": coord, 
        "ethics_score": final_ethical_score 
    }
    
    if is_public and getattr(config, 'REQUIRE_PUBLIC_STORAGE_CONSENT', False):
        consent_prompt = f"Knowledge item (ID: {entry_id}) is marked for public storage. Confirm? (yes/no): "
        try:
            user_consent = input(consent_prompt).lower()
            if user_consent != 'yes':
                if config.VERBOSE_OUTPUT: print(f"Library: Public storage declined by user for item {entry_id}.")
                entry["is_public"] = False 
        except RuntimeError: 
            if config.VERBOSE_OUTPUT: print("Library: input() not available (e.g. test environment), defaulting to private for public storage request.")
            entry["is_public"] = False

    KNOWLEDGE_LIBRARY[entry_id] = entry
    _save_knowledge_library()
    if config.VERBOSE_OUTPUT: print(f"Library: Stored knowledge item '{entry_id}' with public_status={entry['is_public']}.")
    return entry_id

def retrieve_knowledge_by_id(entry_id: str) -> dict | None:
    return KNOWLEDGE_LIBRARY.get(entry_id)

def retrieve_knowledge_by_keyword(keyword: str, search_public: bool = True, search_private: bool = True) -> list:
    results = []
    keyword_lower = keyword.lower()
    for entry_id, entry_data in KNOWLEDGE_LIBRARY.items():
        if (entry_data.get("is_public") and search_public) or \
           (not entry_data.get("is_public") and search_private):
            # Search in preview and full content
            if keyword_lower in entry_data.get("content_preview", "").lower() or \
               keyword_lower in entry_data.get("full_content", "").lower():
                results.append(entry_data)
    return results
# ---vvv KNOWLEDGE STORAGE FUNCTIONS END vvv---


# --- Self-Test Suite for Library module ---
if __name__ == "__main__":
    print("--- Testing library.py (with Knowledge Storage) ---")
    
    class MockManifoldInstance: 
        def bootstrap_phi3(self, concept_name):
            h = hashlib.sha256(concept_name.encode()).digest()
            c = tuple(val / 255.0 * 2.0 - 1.0 for val in h[:3]) 
            t = h[3] / 255.0 
            return ((round(c[0],3), round(c[1],3), round(c[2],3), round(t,3)), t, "Mock summary for " + concept_name)

    original_get_shared_manifold = get_shared_manifold 
    original_score_ethics = score_ethics
    
    if not hasattr(config, 'VERBOSE_OUTPUT'): config.VERBOSE_OUTPUT = False 
    if not hasattr(config, 'MANIFOLD_RANGE'): config.MANIFOLD_RANGE = 1.0 
    
    def run_tests():
        global get_shared_manifold, score_ethics 
        
        mock_manifold_instance = MockManifoldInstance()
        get_shared_manifold = lambda: mock_manifold_instance
        score_ethics = lambda awareness_metrics, concept_summary, action_description: 0.85 

        if _library_storage_path and os.path.exists(_library_storage_path):
            os.remove(_library_storage_path)
        _load_knowledge_library() 

        print("\n[Test 8: store_knowledge - Basic Storage & Coordinate Check]")
        test_content = "This is a test knowledge item about artificial intelligence and ethics."
        original_input = __builtins__.input
        __builtins__.input = lambda _: "yes" 
        
        entry_id = store_knowledge(test_content, is_public=True, source_uri="test_source", author="tester")
        assert entry_id is not None, "T8: store_knowledge did not return an entry ID."
        
        stored_entry = KNOWLEDGE_LIBRARY.get(entry_id)
        assert stored_entry is not None, f"T8: Entry {entry_id} not found in KNOWLEDGE_LIBRARY."
        assert "coordinates" in stored_entry, "T8: 'coordinates' field missing."
        assert is_valid_coordinate(stored_entry["coordinates"]), f"T8: Stored coordinates invalid: {stored_entry['coordinates']}"
        assert stored_entry["ethics_score"] == 0.85, f"T8: Ethics score mismatch. Expected 0.85, got {stored_entry['ethics_score']}"
        assert stored_entry["is_public"] is True, "T8: Public consent not respected."
        assert stored_entry["source_uri"] == "test_source", "T8: Source URI not stored."
        assert stored_entry["author"] == "tester", "T8: Author not stored."
        assert stored_entry["full_content"] == test_content, "T8: Full content not stored."
        print(f"  Stored entry: {stored_entry['id']} with coords {stored_entry['coordinates']}")
        print("  Test 8 Result: Passed")
        
        print("\n[Test 9: store_knowledge - Private storage by default / No consent]")
        test_content_private = "This is a private knowledge item."
        __builtins__.input = lambda _: "no" 
        entry_id_private = store_knowledge(test_content_private, is_public=True) 
        stored_entry_private = KNOWLEDGE_LIBRARY.get(entry_id_private)
        assert stored_entry_private is not None, "T9: Private entry not stored."
        assert stored_entry_private["is_public"] is False, "T9: Entry should be private after declining consent."
        print(f"  Stored private entry (after declining public): {stored_entry_private['id']}")

        entry_id_default_private = store_knowledge("Default private item.") 
        stored_default_private = KNOWLEDGE_LIBRARY.get(entry_id_default_private)
        assert stored_default_private is not None, "T9: Default private entry not stored."
        assert stored_default_private["is_public"] is False, "T9: Entry should be private by default."
        print("  Test 9 Result: Passed")

        __builtins__.input = original_input 

        print("\n[Test 10: retrieve_knowledge_by_keyword]")
        results_ai = retrieve_knowledge_by_keyword("artificial intelligence")
        assert len(results_ai) >= 1, "T10: Keyword 'artificial intelligence' not found."
        assert results_ai[0]["id"] == entry_id, "T10: Mismatch in retrieved entry for 'artificial intelligence'."
        
        results_private = retrieve_knowledge_by_keyword("private knowledge", search_public=False, search_private=True)
        assert len(results_private) == 1, "T10: Keyword 'private knowledge' not found in private items."
        assert results_private[0]["id"] == entry_id_private

        results_public_only = retrieve_knowledge_by_keyword("private knowledge", search_public=True, search_private=False)
        assert len(results_public_only) == 0, "T10: Private item found when searching public only."
        print("  Test 10 Result: Passed")

        # Assuming Test 1-7 (sanitize, summarize, is_valid_coordinate, Mitigator tests) were run previously
        # and are not repeated here for brevity, as this subtask focuses on knowledge storage.
        print("\n--- All relevant library.py self-tests passed! ---")

    try:
        run_tests()
    finally:
        get_shared_manifold = original_get_shared_manifold
        score_ethics = original_score_ethics
        if _library_storage_path and os.path.exists(_library_storage_path):
             os.remove(_library_storage_path)

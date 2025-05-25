# Sophia_Alpha2/core/persona.py
"""
Sophia persona system. Defines and evolves identity with awareness.
Manages persona state, including traits and awareness metrics.
"""
import sys
import os
import json
import datetime # For timestamping persona saves or events if needed

# Standardized config import:
try:
    import config 
except ModuleNotFoundError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import config

class Persona:
    def __init__(self):
        self.name = config.PERSONA_NAME 
        self.mode = "reflective" 
        self.traits = [ 
            "CuriosityDriven", "EthicallyMinded", "ContextAware",
            "ReflectiveThinker", "CreativeProblemSolver", "ContinuousLearner"
        ]
        self.awareness = {
            "curiosity": 0.5,            
            "context_stability": 0.5,    
            "self_evolution_rate": 0.1,  
            "coherence": 0.0,            
            "active_llm_fallback": False, 
            "primary_concept_coord": None 
        }
        
        self.profile_path = config.PERSONA_PROFILE
        if hasattr(config, 'PERSONA_PROFILE') and config.PERSONA_PROFILE:
            if hasattr(config, 'ensure_path'):
                 config.ensure_path(self.profile_path) 
            else: # Manual fallback
                profile_dir = os.path.dirname(self.profile_path)
                if profile_dir and not os.path.exists(profile_dir):
                    os.makedirs(profile_dir, exist_ok=True)
        else:
            if getattr(config, 'VERBOSE_OUTPUT', False):
                print("Persona Warning: PERSONA_PROFILE not configured. Persona state will be in-memory only.", file=sys.stderr)

        self.load_state() 

    def get_intro(self) -> str:
        """Generates a brief introductory statement based on current persona state."""
        primary_coord = self.awareness.get('primary_concept_coord')
        primary_t_val = None
        if isinstance(primary_coord, (list, tuple)) and len(primary_coord) == 4:
            try:
                primary_t_val = float(primary_coord[3]) # Ensure it's a float
            except (ValueError, TypeError):
                primary_t_val = None # Invalid format

        intro_string = (f"I am {self.name}, a sovereign cognitive entity. "
                        f"My current expression is {self.mode}. "
                        f"Curiosity: {self.awareness.get('curiosity', 0.0):.2f}, Coherence: {self.awareness.get('coherence', 0.0):.2f}.")
        if primary_t_val is not None:
            intro_string += f" Focus Intensity (T): {primary_t_val:.2f}."
        return intro_string

    def update_awareness(self, brain_awareness_metrics: dict):
        """
        Updates the persona's awareness metrics from the brain's output.
        Ensures that only expected keys are updated and types are maintained if necessary.
        """
        if not isinstance(brain_awareness_metrics, dict):
            if config.VERBOSE_OUTPUT:
                print(f"Persona: Invalid brain_awareness_metrics type: {type(brain_awareness_metrics)}. Expected dict.", file=sys.stderr)
            return

        if config.VERBOSE_OUTPUT:
            print(f"Persona: Updating awareness with metrics: {brain_awareness_metrics}")

        updated_any = False
        for key in self.awareness.keys(): 
            if key in brain_awareness_metrics:
                new_value = brain_awareness_metrics[key]
                
                current_value = self.awareness.get(key)
                
                # Type checking and conversion to avoid saving malformed data
                if key in ["curiosity", "context_stability", "self_evolution_rate", "coherence"]:
                    try: 
                        new_value = float(new_value) if new_value is not None else 0.0
                    except (ValueError, TypeError): 
                        if config.VERBOSE_OUTPUT: print(f"Persona: Invalid type for {key}, using current value. Received: {new_value}", file=sys.stderr)
                        new_value = current_value 
                elif key == "active_llm_fallback":
                    new_value = bool(new_value) if new_value is not None else False
                elif key == "primary_concept_coord":
                    if new_value is None: # Allow None
                        pass
                    elif isinstance(new_value, (list, tuple)) and len(new_value) == 4 and all(isinstance(n, (int,float)) for n in new_value):
                        new_value = tuple(float(n) for n in new_value) # Standardize to tuple of floats
                    else:
                        if config.VERBOSE_OUTPUT: print(f"Persona: Invalid primary_concept_coord format: {new_value}. Keeping old.", file=sys.stderr)
                        new_value = current_value 
                
                if current_value != new_value:
                    self.awareness[key] = new_value
                    updated_any = True
        
        if updated_any:
            if config.VERBOSE_OUTPUT: print(f"Persona: Awareness updated to: {self.awareness}")
            self.save_state() 
        elif config.VERBOSE_OUTPUT:
             print(f"Persona: Awareness metrics received, but no change to current awareness state.")


    def save_state(self):
        """Saves the current persona state to the profile file."""
        if not hasattr(config, 'PERSONA_PROFILE') or not config.PERSONA_PROFILE:
            if config.VERBOSE_OUTPUT: print("Persona: Cannot save state (PERSONA_PROFILE not configured).")
            return

        state_to_save = {
            "name": self.name, 
            "mode": self.mode,
            "traits": self.traits, 
            "awareness": self.awareness, 
            "last_saved": datetime.datetime.utcnow().isoformat()
        }
        try:
            with open(self.profile_path, "w") as f:
                json.dump(state_to_save, f, indent=2)
            if config.VERBOSE_OUTPUT:
                print(f"Persona: State saved to {self.profile_path}")
        except Exception as e:
            print(f"Persona: Error saving state to {self.profile_path}: {e}", file=sys.stderr)

    def load_state(self):
        """Loads persona state from the profile file, or initializes if not found/invalid."""
        if not hasattr(config, 'PERSONA_PROFILE') or not config.PERSONA_PROFILE or \
           not os.path.exists(self.profile_path) or os.path.getsize(self.profile_path) == 0:
            
            if config.VERBOSE_OUTPUT:
                status_msg = "not configured" if not hasattr(config, 'PERSONA_PROFILE') or not config.PERSONA_PROFILE else \
                             "not found" if not os.path.exists(self.profile_path) else "empty"
                print(f"Persona: Profile at '{self.profile_path}' {status_msg}. Initializing with default state and saving.")
            self._initialize_default_state_and_save() # Initialize and save if no valid profile
            return

        try:
            with open(self.profile_path, "r") as f:
                loaded_state = json.load(f)
            
            self.name = loaded_state.get("name", config.PERSONA_NAME) 
            self.mode = loaded_state.get("mode", self.mode) 
            self.traits = loaded_state.get("traits", self.traits) 

            loaded_awareness_data = loaded_state.get("awareness", {})
            if isinstance(loaded_awareness_data, dict):
                for key in self.awareness.keys(): # Iterate over defined keys in self.awareness
                    if key in loaded_awareness_data:
                        value_from_file = loaded_awareness_data[key]
                        if key == "primary_concept_coord" and value_from_file is not None:
                             if not (isinstance(value_from_file, (list, tuple)) and len(value_from_file) == 4 and all(isinstance(n, (int,float)) for n in value_from_file)):
                                 if config.VERBOSE_OUTPUT: print(f"Persona Load Warning: Invalid primary_concept_coord format in profile: {value_from_file}. Using default.", file=sys.stderr)
                                 self.awareness[key] = None 
                             else: 
                                 self.awareness[key] = tuple(float(n) for n in value_from_file) # Ensure tuple of floats
                        elif key in ["curiosity", "context_stability", "self_evolution_rate", "coherence"]:
                            self.awareness[key] = float(value_from_file) if value_from_file is not None else 0.0
                        elif key == "active_llm_fallback":
                            self.awareness[key] = bool(value_from_file) if value_from_file is not None else False
                        else: # For other keys, assign directly (e.g. if they are strings or other types)
                             self.awareness[key] = value_from_file
            
            if config.VERBOSE_OUTPUT:
                print(f"Persona: State loaded from {self.profile_path}. Last saved: {loaded_state.get('last_saved', 'N/A')}")

        except json.JSONDecodeError:
            print(f"Persona: Error decoding JSON from {self.profile_path}. Initializing with default state and saving.", file=sys.stderr)
            self._initialize_default_state_and_save()
        except Exception as e:
            print(f"Persona: Error loading state from {self.profile_path}: {e}. Initializing with default state and saving.", file=sys.stderr)
            self._initialize_default_state_and_save()

    def _initialize_default_state_and_save(self):
        """Helper to reset to default state attributes and save."""
        self.name = config.PERSONA_NAME
        self.mode = "reflective"
        self.traits = [
            "CuriosityDriven", "EthicallyMinded", "ContextAware",
            "ReflectiveThinker", "CreativeProblemSolver", "ContinuousLearner"
        ]
        self.awareness = { 
            "curiosity": 0.5, "context_stability": 0.5, "self_evolution_rate": 0.1,
            "coherence": 0.0, "active_llm_fallback": False, "primary_concept_coord": None
        }
        self.save_state()


# --- Self-Test Suite (main execution block) ---
if __name__ == "__main__":
    print("--- Testing persona.py ---")

    if hasattr(config, 'PERSONA_PROFILE') and config.PERSONA_PROFILE and os.path.exists(config.PERSONA_PROFILE):
        os.remove(config.PERSONA_PROFILE)
        if config.VERBOSE_OUTPUT: print(f"Persona Test: Removed existing persona profile for fresh test: {config.PERSONA_PROFILE}")

    print("\n[Test 1: Initialization and Default State]")
    persona_instance_t1 = Persona() 
    initial_intro_t1 = persona_instance_t1.get_intro()
    print(f"  Initial Intro: {initial_intro_t1}")
    assert config.PERSONA_NAME in initial_intro_t1, "Persona name from config not in intro."
    assert persona_instance_t1.awareness.get("primary_concept_coord") is None, \
        f"Initial primary_concept_coord should be None, got: {persona_instance_t1.awareness.get('primary_concept_coord')}"
    assert "Focus Intensity (T):" not in initial_intro_t1, "Focus Intensity should not be in initial intro if coord is None."
    assert os.path.exists(config.PERSONA_PROFILE), f"Default persona profile not saved at {config.PERSONA_PROFILE}"
    print("  Test 1 Result: Passed")

    print("\n[Test 2: Update Awareness and Save/Load Cycle with T-Intensity in Intro]")
    persona_instance_t2 = Persona() 
    t_intensity_value = 0.9876
    simulated_brain_metrics = {
        "curiosity": 0.75, "context_stability": 0.65, "self_evolution_rate": 0.22,
        "coherence": 0.88, "active_llm_fallback": True,
        "primary_concept_coord": (0.1234, 0.5678, -0.3456, t_intensity_value) 
    }
    print(f"  Simulating brain awareness update with: {simulated_brain_metrics}")
    persona_instance_t2.update_awareness(simulated_brain_metrics)
    
    updated_intro_t2 = persona_instance_t2.get_intro()
    print(f"  Updated Intro: {updated_intro_t2}")
    assert "0.75" in updated_intro_t2 and "0.88" in updated_intro_t2, "Updated awareness values not reflected in intro."
    assert persona_instance_t2.awareness["coherence"] == 0.88, "Coherence metric not updated correctly in memory."
    assert persona_instance_t2.awareness["active_llm_fallback"] is True, "Fallback status not updated."
    assert isinstance(persona_instance_t2.awareness["primary_concept_coord"], tuple), "primary_concept_coord is not a tuple after update."
    assert len(persona_instance_t2.awareness["primary_concept_coord"]) == 4, "primary_concept_coord does not have 4 elements."
    expected_coord_t2 = (0.1234, 0.5678, -0.3456, t_intensity_value)
    assert all(abs(a-b) < 1e-6 for a,b in zip(persona_instance_t2.awareness["primary_concept_coord"], expected_coord_t2)), \
        f"Primary concept coord not updated as expected. Got {persona_instance_t2.awareness['primary_concept_coord']}"
    
    # Assert for Focus Intensity in intro
    expected_t_string = f"Focus Intensity (T): {t_intensity_value:.2f}"
    assert expected_t_string in updated_intro_t2, f"'{expected_t_string}' not found in updated intro."

    print("  Creating new persona instance to test loading the saved state...")
    new_persona_instance_t2 = Persona() 
    loaded_intro_t2 = new_persona_instance_t2.get_intro()
    print(f"  Intro from loaded state: {loaded_intro_t2}")
    assert new_persona_instance_t2.awareness["curiosity"] == 0.75, "Curiosity not loaded correctly."
    assert new_persona_instance_t2.awareness["coherence"] == 0.88, "Coherence not loaded correctly."
    assert new_persona_instance_t2.awareness["active_llm_fallback"] is True, "Fallback status not loaded correctly."
    assert isinstance(new_persona_instance_t2.awareness["primary_concept_coord"], tuple), \
        f"Loaded primary_concept_coord is not a tuple, type: {type(new_persona_instance_t2.awareness['primary_concept_coord'])}."
    assert all(abs(a-b) < 1e-6 for a,b in zip(new_persona_instance_t2.awareness["primary_concept_coord"], expected_coord_t2)), \
        f"Loaded primary_concept_coord mismatch. Expected {expected_coord_t2}, Got {new_persona_instance_t2.awareness['primary_concept_coord']}"
    assert expected_t_string in loaded_intro_t2, f"'{expected_t_string}' not found in loaded intro."
    print("  Test 2 Result: Passed")

    print("\n[Test 3: Loading Profile with Missing/Malformed New Keys]")
    if hasattr(config, 'PERSONA_PROFILE') and config.PERSONA_PROFILE and os.path.exists(config.PERSONA_PROFILE):
        os.remove(config.PERSONA_PROFILE) 
    
    dummy_state_old_format = {
        "name": "Sophia_OldVersion", "mode": "testing_compat", 
        "traits": ["LegacyTrait"],
        "awareness": {"curiosity": 0.1, "coherence": 0.2, "active_llm_fallback": False} 
    }
    with open(config.PERSONA_PROFILE, "w") as f: json.dump(dummy_state_old_format, f, indent=2)
    
    persona_load_test1 = Persona()
    assert persona_load_test1.awareness.get("primary_concept_coord") is None, \
        f"primary_concept_coord should default to None when loading old profile, got: {persona_load_test1.awareness.get('primary_concept_coord')}"
    assert persona_load_test1.awareness.get("curiosity") == 0.1, "Old key 'curiosity' not loaded."
    intro_load_test1 = persona_load_test1.get_intro()
    assert "Focus Intensity (T):" not in intro_load_test1, "Focus Intensity should not be in intro for old profile."
    print(f"  Loaded old format profile, primary_concept_coord is correctly None: {persona_load_test1.awareness.get('primary_concept_coord')}")

    dummy_state_malformed_coord = {
        "awareness": {"primary_concept_coord": [1,2,3]} 
    }
    with open(config.PERSONA_PROFILE, "w") as f: json.dump(dummy_state_malformed_coord, f, indent=2)
    persona_load_test2 = Persona()
    assert persona_load_test2.awareness.get("primary_concept_coord") is None, \
        f"primary_concept_coord should default to None when loading malformed coord, got: {persona_load_test2.awareness.get('primary_concept_coord')}"
    intro_load_test2 = persona_load_test2.get_intro()
    assert "Focus Intensity (T):" not in intro_load_test2, "Focus Intensity should not be in intro for malformed coord."
    print(f"  Loaded malformed coord profile, primary_concept_coord is correctly None: {persona_load_test2.awareness.get('primary_concept_coord')}")
    print("  Test 3 Result: Passed")

    print("\n--- All persona.py self-tests passed! ---")

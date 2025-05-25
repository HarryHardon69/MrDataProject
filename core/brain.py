# Sophia_Alpha2/core/brain.py
"""
Sophia's cognitive core with snnTorch-based spacetime manifold,
enhanced with LLM (e.g., Gemma via LM Studio or Phi-3 via API) bootstrapping and connectivity.
"""
import numpy as np
import torch
import snntorch as snn
from snntorch import surrogate
import json
import requests 
import re
import socket
import sys
import os
from datetime import datetime

# Standardized config import:
try:
    import config 
except ModuleNotFoundError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import config

# Ensure log path from config is ready
if hasattr(config, 'SYSTEM_LOG_PATH') and config.SYSTEM_LOG_PATH:
    if hasattr(config, 'ensure_path'): # Check if ensure_path exists in the config module
        config.ensure_path(config.SYSTEM_LOG_PATH) 
    else: # Manual fallback if ensure_path is not available
        log_dir = os.path.dirname(config.SYSTEM_LOG_PATH)
        if log_dir and not os.path.exists(log_dir): # Ensure directory exists
            os.makedirs(log_dir, exist_ok=True)
else:
    # Use getattr to safely access VERBOSE_OUTPUT, defaulting to False if not set
    if getattr(config, 'VERBOSE_OUTPUT', False):
        print("Brain Warning: SYSTEM_LOG_PATH not found/configured in config.py. Logging may fail.", file=sys.stderr)


_shared_manifold_instance = None

class SpacetimeManifold:
    def __init__(self):
        self.range = getattr(config, 'MANIFOLD_RANGE', 1.0)
        resource_profile = getattr(config, 'RESOURCE_PROFILE', {"MAX_NEURONS": 1000, "RESOLUTION": 3, "SNN_TIME_STEPS": 20})
        self.neurons = int(resource_profile.get("MAX_NEURONS", 1000)) 
        self.resolution_decimals = resource_profile.get("RESOLUTION", 3)
        self.batch_size = 1 
        self.num_steps = resource_profile.get("SNN_TIME_STEPS", 20) 
        self.coordinates = {} 
        self.input_size = getattr(config, 'SNN_INPUT_SIZE', 10) 
        
        # STDP parameters (tunable via config)
        self.tau_stdp = getattr(config, 'STDP_WINDOW_MS', 20.0) / 1000.0 # Time constant for STDP window (s)
        self.lr_stdp = getattr(config, 'HEBBIAN_LEARNING_RATE', 1.0 / np.sqrt(self.neurons if self.neurons > 0 else 1)) # Learning rate for STDP

        self.coherence = 0.0 
        self.last_avg_stdp_weight_change = 0.0 

        self.fc = torch.nn.Linear(self.input_size, self.neurons)
        snn_slope = getattr(config, 'SNN_SURROGATE_SLOPE', 25.0) # Tunable surrogate gradient slope
        spike_grad_surrogate = surrogate.fast_sigmoid(slope=snn_slope) 
        
        # Tunable LIF neuron parameters
        lif_beta = getattr(config, 'SNN_LIF_BETA', 0.9) # Membrane potential decay rate
        lif_threshold = getattr(config, 'SNN_LIF_THRESHOLD', 0.7) # Firing threshold
        self.lif1 = snn.Leaky(beta=lif_beta, threshold=lif_threshold, spike_grad=spike_grad_surrogate, learn_beta=True)
        
        self.mem = self.lif1.init_leaky() 
        self.spk = self.lif1.init_leaky() 
        
        snn_optimizer_lr = getattr(config, 'SNN_OPTIMIZER_LR', 0.01) # Tunable SNN optimizer learning rate
        self.optimizer = torch.optim.Adam(list(self.fc.parameters()) + list(self.lif1.parameters()), lr=snn_optimizer_lr)

        # Step 4: Add _log_system_event calls - Manifold Initialization
        self._log_system_event("manifold_initialized", {
            "range": self.range, "neurons": self.neurons, "resolution": self.resolution_decimals,
            "snn_steps": self.num_steps, "input_size": self.input_size,
            "tau_stdp_ms": self.tau_stdp * 1000, "lr_stdp": self.lr_stdp,
            "lif_beta": self.lif1.beta.item() if isinstance(self.lif1.beta, torch.Tensor) else self.lif1.beta,
            "lif_threshold": self.lif1.threshold, "snn_slope": snn_slope, "optimizer_lr": snn_optimizer_lr
        })
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT: # Safe check for VERBOSE_OUTPUT
            print(f"Brain: SpacetimeManifold initialized. Neurons={self.neurons}, STDP Tau={self.tau_stdp*1000:.1f}ms, LIF Beta={lif_beta:.2f}")

    def _log_system_event(self, event_type, data):
        log_path = getattr(config, 'SYSTEM_LOG_PATH', None)
        verbose_output = hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT
        if not log_path:
            if verbose_output: print(f"Brain SystemLog (NoPath): {event_type} - {data}")
            return
        try:
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
            # Ensure all data is JSON serializable
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    serializable_data[key] = value
                elif isinstance(value, (np.ndarray, torch.Tensor)): # Convert tensors/arrays
                    serializable_data[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
                else:
                    serializable_data[key] = str(value) # Fallback to string

            log_entry = {"timestamp": datetime.utcnow().isoformat(), "module": "brain", "event_type": event_type, "data": serializable_data}
            with open(log_path, "a") as f:
                json.dump(log_entry, f); f.write("\n")
        except Exception as e:
            if verbose_output: print(f"Brain SystemLog Error: {event_type}: {e}", file=sys.stderr)

    def _mock_phi3_concept_data(self, concept_name): 
        concept_name_lower = concept_name.lower().strip()
        effective_concept_name = concept_name if concept_name_lower else 'the_void_of_emptiness' 
        default_summary = f"Reflecting deeply on the concept of '{effective_concept_name}'... It evokes a sense of {'mystery' if concept_name_lower else 'vastness'}, prompting further inquiry into its manifold connections and implications within the cognitive architecture."
        mock_db = {
            "love": {"summary": "Love is a complex, profound, and multifaceted emotional bond and experience...", "valence": 0.8, "abstraction": 0.7, "relevance": 0.95, "intensity": 0.9},
            "ethics": {"summary": "Ethics involves systematizing, defending, and recommending concepts of right and wrong conduct...", "valence": 0.1, "abstraction": 0.8, "relevance": 0.9, "intensity": 0.6},
            "phi-3": {"summary": "Phi-3 is a family of small language models developed by Microsoft, designed to be efficient and capable for various tasks.", "valence": 0.5, "abstraction": 0.6, "relevance": 0.8, "intensity": 0.7},
            "unknown": {"summary": "The 'unknown' represents that which is not yet discovered...", "valence": -0.1, "abstraction": 0.6, "relevance": 0.7, "intensity": 0.7},
            "the_void_of_emptiness": {"summary": default_summary, "valence": -0.2, "abstraction": 0.9, "relevance": 0.5, "intensity": 0.8}
        }
        data = mock_db.get(concept_name_lower, {
            "summary": default_summary, "valence": 0.0, "abstraction": 0.5, "relevance": 0.5, "intensity": 0.5 
        })
        self._log_system_event("mock_phi3_data_used", {"concept": concept_name, "data_source": "internal_mock_db"})
        if config.VERBOSE_OUTPUT: print(f"Brain: Using mock Phi-3 data for '{concept_name}': {data}")
        return data

    def _try_connect_phi3(self): 
        llm_base_url = getattr(config, 'LLM_BASE_URL', None)
        if not llm_base_url or llm_base_url == "not_set":
            if config.VERBOSE_OUTPUT: print("Brain LLM Check: LLM_BASE_URL not configured.")
            return False
        if not ("localhost" in llm_base_url or "127.0.0.1" in llm_base_url):
            if config.VERBOSE_OUTPUT: print("Brain LLM Check: Non-localhost URL, assuming reachable.")
            return True 
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(llm_base_url)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80) 
            if "ollama" in getattr(config, 'LLM_PROVIDER', "").lower() or ("11434" in llm_base_url): port = 11434
            elif "lm_studio" in getattr(config, 'LLM_PROVIDER', "").lower() or ("1234" in llm_base_url): port = 1234
            with socket.create_connection((host, port), timeout=getattr(config, 'LLM_CONNECTION_TIMEOUT', 2)):
                if config.VERBOSE_OUTPUT: print(f"Brain: LLM server connection check to {host}:{port} successful.")
                return True
        except Exception as e:
            if config.VERBOSE_OUTPUT: print(f"Brain: LLM server connection check to {llm_base_url} (host: {host}, port: {port}) failed: {e}", file=sys.stderr)
            return False

    # Step 2: Integrate Real Phi-3 API Call Logic
    def bootstrap_phi3(self, concept_name: str):
        self._log_system_event("bootstrap_phi3_start", {"concept": concept_name, "provider_configured": config.LLM_PROVIDER})
        concept_data = None; llm_error_details = None; used_real_api = False
        
        enable_api_flag = getattr(config, 'ENABLE_LLM_API', False) 

        if enable_api_flag and self._try_connect_phi3():
            try:
                from openai import OpenAI 
                api_key_to_use = getattr(config, 'LLM_API_KEY', 'None')
                client = OpenAI(base_url=config.LLM_BASE_URL, api_key=api_key_to_use)
                
                system_prompt = config.LLM_CONCEPT_PROMPT_TEMPLATE
                user_prompt_content = f"Concept: '{concept_name}'"
                prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_content}]
                
                completion_args = {
                    "model": config.LLM_MODEL, 
                    "messages": prompt_messages,
                    "temperature": config.LLM_TEMPERATURE,
                    "max_tokens": 250, 
                    "response_format": {"type": "json_object"} 
                }
                self._log_system_event("phi3_api_call_attempt", {"concept": concept_name, "args_summary": {"model": config.LLM_MODEL, "temp": config.LLM_TEMPERATURE}})
                if config.VERBOSE_OUTPUT: print(f"Brain: Attempting real Phi-3 API call for '{concept_name}' to {config.LLM_BASE_URL}")
                
                completion = client.chat.completions.create(**completion_args)
                llm_response_text = completion.choices[0].message.content.strip()
                used_real_api = True
                self._log_system_event("phi3_api_call_success", {"concept": concept_name, "response_snippet": llm_response_text[:70]})
                if config.VERBOSE_OUTPUT: print(f"Brain: Phi-3 API response for '{concept_name}': {llm_response_text[:100]}...")

                try:
                    processed_text = llm_response_text
                    if processed_text.startswith("```json"): processed_text = processed_text[7:]
                    if processed_text.endswith("```"): processed_text = processed_text[:-3]
                    processed_text = processed_text.strip()
                    concept_data = json.loads(processed_text)
                    required_keys = ["summary", "valence", "abstraction", "relevance", "intensity"]
                    if not all(k in concept_data for k in required_keys):
                        raise ValueError("Missing required keys in Phi-3 JSON response.")
                    for k_num in required_keys[1:]: concept_data[k_num] = float(concept_data[k_num])
                    self._log_system_event("phi3_response_parsed", {"concept": concept_name})
                except (json.JSONDecodeError, ValueError, TypeError) as e_parse:
                    llm_error_details = f"Phi-3 response parsing failed: {e_parse}. Response: '{llm_response_text[:200]}...'"
                    self._log_system_event("phi3_response_parse_error", {"concept": concept_name, "error": llm_error_details})
                    concept_data = None 
            
            except Exception as e_api:
                llm_error_details = f"Phi-3 API call failed: {type(e_api).__name__}: {e_api}."
                self._log_system_event("phi3_api_call_error", {"concept": concept_name, "error": llm_error_details})
                concept_data = None 

        if concept_data is None: 
            fallback_reason = llm_error_details or ("API disabled" if not enable_api_flag else "Connection/other API error")
            self._log_system_event("bootstrap_phi3_fallback_mock", {"concept": concept_name, "reason": fallback_reason})
            if config.VERBOSE_OUTPUT: print(f"Brain: Using mock Phi-3 data for '{concept_name}' due to: {fallback_reason}")
            concept_data = self._mock_phi3_concept_data(concept_name)

        x = self.range * np.clip(float(concept_data.get("valence", 0.0)), -1.0, 1.0)
        y = self.range * np.clip(float(concept_data.get("abstraction", 0.0)), 0.0, 1.0)
        z = self.range * np.clip(float(concept_data.get("relevance", 0.0)), 0.0, 1.0)  
        t_intensity = np.clip(float(concept_data.get("intensity", 0.0)), 0.0, 1.0)   

        coord = (round(x, self.resolution_decimals), round(y, self.resolution_decimals), 
                 round(z, self.resolution_decimals), round(t_intensity, self.resolution_decimals))
        
        self.coordinates[concept_name] = coord
        self._log_system_event("bootstrap_phi3_end", {"concept": concept_name, "coordinates": coord, "used_real_api": used_real_api, "final_data_source": "mock" if not used_real_api or llm_error_details else "api"})
        return coord, concept_data.get("intensity", 0.0), concept_data.get("summary", "Summary not available.")

    # Step 5: Review STDP and Spiking Code
    def update_stdp(self, pre_spk_flat, post_spk_flat, weights, concept_name, prev_t_intensity, dt_threshold=0.01):
        pre_spk, post_spk = pre_spk_flat[0], post_spk_flat[0] 
        delta_w = torch.zeros_like(weights) 
        current_concept_coord = self.coordinates.get(concept_name)
        if not current_concept_coord:
            if config.VERBOSE_OUTPUT: print(f"Brain STDP Warning: Concept '{concept_name}' not found for STDP.", file=sys.stderr)
            return weights, 0.0 
        current_t_intensity = current_concept_coord[3] 
        delta_t = abs(current_t_intensity - (prev_t_intensity if prev_t_intensity is not None else current_t_intensity))
        exp_term = np.exp(-delta_t / self.tau_stdp) # tau_stdp from config
        for i in range(self.neurons): 
            if post_spk[i] > 0: 
                for j in range(self.input_size): 
                    if pre_spk[j] > 0: 
                        if delta_t < dt_threshold: # dt_threshold is a fixed parameter here
                            delta_w[i, j] += self.lr_stdp * exp_term # lr_stdp from config
                        else: 
                            # STDP_DEPRESSION_FACTOR from config controls relative strength of LTD
                            delta_w[i, j] -= self.lr_stdp * getattr(config, 'STDP_DEPRESSION_FACTOR', 0.1) * exp_term 
        weights_updated = weights + delta_w
        delta_w_mean_abs = delta_w.abs().mean().item()
        if delta_w_mean_abs > 1e-7: # Threshold for significant change
            # COHERENCE_UPDATE_FACTOR from config scales STDP impact on coherence
            coherence_change = getattr(config, 'COHERENCE_UPDATE_FACTOR', 0.1) * delta_w_mean_abs * exp_term 
            self.coherence = min(1.0, max(0.0, self.coherence + coherence_change)) 
            self._log_system_event("stdp_update_applied", {
                "concept": concept_name, "delta_t_sec": delta_t, "exp_term": exp_term,
                "delta_w_mean_abs": delta_w_mean_abs, "new_coherence": self.coherence
            })
        return weights_updated, delta_w_mean_abs

    def warp_manifold(self, input_text: str):
        self._log_system_event("snn_warp_start", {"input_length": len(input_text), "concept_preview": input_text[:50]})
        thought_steps_log = [f"Brain: Initializing SNN warp for '{input_text[:50]}...'"]
        generated_monologue_parts = [] 
        primary_concept_name = input_text.strip() if input_text.strip() else "silence_or_void"
        coord, intensity, summary = self.bootstrap_phi3(primary_concept_name) 
        current_concept_coord_for_warp = coord 
        generated_monologue_parts.append(f"Reflecting on '{primary_concept_name}': {summary}")
        generated_monologue_parts.append(f"Manifold position for '{primary_concept_name}': {coord} (Intensity: {intensity:.2f}).")
        all_spk_rec, all_mem_rec, t_intensity_diffs = [], [], [] 
        total_stdp_weight_change_metric_this_warp = 0.0; num_stdp_updates_this_warp = 0
        current_weights = self.fc.weight.clone().detach() 
        self.mem, self.spk = self.lif1.init_leaky(), self.lif1.init_leaky() 
        prev_concept_t_intensity = None 
        for step in range(self.num_steps): # num_steps from config
            input_features = torch.zeros(self.batch_size, self.input_size) 
            input_features[0, 0] = 0.5 + float(intensity) # SNN input modulation example
            input_features = torch.clamp(input_features, 0, 1.0) 
            current_fc_out = self.fc(input_features) 
            spk_out, mem_out = self.lif1(current_fc_out, self.mem) # LIF neuron parameters from config
            all_spk_rec.append(spk_out.clone().detach()); all_mem_rec.append(mem_out.clone().detach())
            self.mem, self.spk = mem_out, spk_out 
            updated_weights, weight_change_metric = self.update_stdp(input_features.detach(), spk_out.detach(), current_weights, primary_concept_name, prev_concept_t_intensity)
            current_weights = updated_weights 
            if weight_change_metric > 1e-7: 
                total_stdp_weight_change_metric_this_warp += weight_change_metric
                num_stdp_updates_this_warp += 1
            spike_count_this_step = spk_out.sum().item()
            thought_steps_log.append(f"WarpStep {step+1}/{self.num_steps}: '{primary_concept_name}'. Spikes={spike_count_this_step:.0f}. Δw_abs={weight_change_metric:.3e}. Coh={self.coherence:.3f}.")
            self._log_system_event("snn_warp_step", {"step": step+1, "total_steps": self.num_steps, "concept": primary_concept_name, "spikes": spike_count_this_step, "stdp_metric": weight_change_metric, "coherence": self.coherence})
            current_t_val = self.coordinates[primary_concept_name][3] 
            if prev_concept_t_intensity is not None: t_intensity_diffs.append(abs(current_t_val - prev_concept_t_intensity))
            prev_concept_t_intensity = current_t_val
        with torch.no_grad(): self.fc.weight.copy_(current_weights)
        self.last_avg_stdp_weight_change = (total_stdp_weight_change_metric_this_warp / num_stdp_updates_this_warp) if num_stdp_updates_this_warp > 0 else 0.0
        final_monologue = "\n".join(generated_monologue_parts)
        if not final_monologue.strip(): final_monologue = f"Completed SNN warp for '{primary_concept_name}'. Coherence: {self.coherence:.3f}."
        self._log_system_event("snn_warp_end", { 
            "input": input_text, "concept": primary_concept_name, 
            "avg_stdp_change": self.last_avg_stdp_weight_change, 
            "num_stdp_updates": num_stdp_updates_this_warp, "final_coherence": self.coherence,
            "total_steps_run": self.num_steps
        })
        return thought_steps_log, final_monologue, all_spk_rec, t_intensity_diffs, current_concept_coord_for_warp

    # Step 3: Enhance Awareness Metrics in think()
    def think(self, input_text: str, stream_thought_steps: bool = False):
        self._log_system_event("think_start", {"input_length": len(input_text), "streaming": stream_thought_steps})
        active_llm_fallback = False 
        primary_concept_coord = None 
        snn_coherence = self.coherence 
        snn_self_evolution_rate = self.last_avg_stdp_weight_change * 1000.0 
        snn_context_stability = 0.5 
        snn_curiosity = 0.3 

        enable_snn_flag = getattr(config, 'ENABLE_SNN', True) 

        if not enable_snn_flag: 
            if config.VERBOSE_OUTPUT: print("Brain: SNN disabled. Using Phi-3 fallback.")
            self._log_system_event("snn_disabled_fallback_phi3", {"input": input_text})
            effective_input = input_text if input_text.strip() else 'probing_consciousness_state' 
            primary_concept_coord, _, summary_from_llm = self.bootstrap_phi3(effective_input) 
            response_text = f"LLM monologue (SNN Bypass): {summary_from_llm}"
            thought_steps = ["SNN processing disabled.", f"LLM reflected on '{effective_input}' at {primary_concept_coord}.", response_text]
            active_llm_fallback = True 
            snn_coherence = 0.0 
            snn_self_evolution_rate = 0.0
        else: 
            try:
                thought_steps, response_text, spk_rec_hist, t_diffs, primary_coord_val = self.warp_manifold(input_text)
                primary_concept_coord = primary_coord_val 
                snn_coherence = self.coherence 
                snn_self_evolution_rate = min(1.0, max(0.0, self.last_avg_stdp_weight_change * 1000.0))
                snn_context_stability = 1.0 - float(np.mean(t_diffs) if t_diffs else 0.0)
                mean_spikes = 0.0
                if spk_rec_hist and self.neurons > 0 and self.num_steps > 0:
                    total_spikes = torch.stack(spk_rec_hist).sum().item()
                    mean_spikes = total_spikes / (len(spk_rec_hist) * self.neurons) 
                max_rate = getattr(config, 'RESOURCE_PROFILE', {}).get("MAX_SPIKE_RATE", 0.1)
                norm_activity = mean_spikes / (max_rate * 0.2 + 1e-6) 
                snn_curiosity = min(1.0, max(0.0, (snn_coherence + norm_activity) * 0.5))
                active_llm_fallback = False 
            except Exception as e:
                self._log_system_event("snn_warp_error_critical", {"input": input_text, "error_type": type(e).__name__, "error_message": str(e)})
                if config.VERBOSE_OUTPUT: print(f"Brain: Critical Error during SNN warp: {e}. Phi-3 fallback.", file=sys.stderr)
                if config.VERBOSE_OUTPUT: import traceback; traceback.print_exc(file=sys.stderr) 
                effective_input_err = input_text if input_text.strip() else 'system_error_reflection'
                primary_concept_coord, _, summary_from_llm = self.bootstrap_phi3(effective_input_err) 
                response_text = f"LLM monologue (SNN Error Recovery): After challenge with '{input_text[:30]}...', reflection: {summary_from_llm}"
                thought_steps = [f"SNN error: {str(e)[:100]}...", "LLM recovery...", response_text]
                active_llm_fallback = True 
                snn_coherence = 0.1; snn_self_evolution_rate = 0.0; snn_context_stability = 0.0; snn_curiosity = 0.7
        
        awareness_metrics = {
            "curiosity": float(snn_curiosity),
            "context_stability": float(snn_context_stability),
            "self_evolution_rate": float(snn_self_evolution_rate),
            "coherence": float(snn_coherence),
            "active_llm_fallback": active_llm_fallback, 
            "primary_concept_coord": primary_concept_coord 
        }
        self._log_system_event("think_completed", {"input_length": len(input_text), "awareness_metrics": awareness_metrics})

        if stream_thought_steps and config.VERBOSE_OUTPUT:
            print("\n--- Brain Thought Stream ---", flush=True); [print(s, flush=True) for s in thought_steps]; print("--- End Stream ---\n", flush=True)
        return thought_steps, response_text, awareness_metrics

def get_shared_manifold():
    global _shared_manifold_instance
    if _shared_manifold_instance is None:
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT: print("Brain: Creating shared SpacetimeManifold instance.")
        _shared_manifold_instance = SpacetimeManifold()
    return _shared_manifold_instance

def think(input_text: str, stream: bool = False):
    manifold = get_shared_manifold()
    return manifold.think(input_text, stream_thought_steps=stream)

if __name__ == "__main__":
    print("--- Testing brain.py (Enhanced with Phi-3 logic & Full Awareness Metrics) ---")
    class TempConfigOverride: 
        def __init__(self, **kwargs):
            self.original_values = {}
            self.config_module_obj = sys.modules.get('config') 
            if not self.config_module_obj: raise RuntimeError("Config module not found for TempConfigOverride.")
            for key, value in kwargs.items():
                self.original_values[key] = getattr(self.config_module_obj, key, None)
                setattr(self.config_module_obj, key, value)
        def __enter__(self): return self
        def __exit__(self, type, value, traceback):
            for key, original_value in self.original_values.items():
                setattr(self.config_module_obj, key, original_value)
            if hasattr(config, 'SYSTEM_LOG_PATH') and config.SYSTEM_LOG_PATH and hasattr(config, 'ensure_path'):
                 config.ensure_path(config.SYSTEM_LOG_PATH)

    test_results_summary = []
    import traceback 
    def reset_manifold_for_test():
        global _shared_manifold_instance
        _shared_manifold_instance = None
        if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT: print("\nBrain Test Helper: Resetting shared manifold instance.")
    
    expected_awareness_keys = ["curiosity", "context_stability", "self_evolution_rate", "coherence", "active_llm_fallback", "primary_concept_coord"]
    print("\n[Test 1: Standard SNN Input with Phi-3 Bootstrap (mocked)]")
    try:
        reset_manifold_for_test()
        with TempConfigOverride(ENABLE_SNN=True, ENABLE_LLM_API=True, VERBOSE_OUTPUT=False, LLM_PROVIDER="mock_for_snn_test", LLM_MODEL="phi-3-mock"):
            thought_steps, response, awareness = think("What is love?", stream=False)
        assert len(thought_steps) > 1, "T1: Not enough thought steps."
        assert "Reflecting on" in response or "SNN warp" in response, "T1: Response format unexpected."
        assert awareness.get("coherence", -1.0) >= 0.0, f"T1: Coherence invalid: {awareness.get('coherence')}"
        assert not awareness.get("active_llm_fallback", True), "T1: LLM fallback should be False for SNN path."
        assert awareness.get("primary_concept_coord") is not None, "T1: Primary concept coord missing."
        assert all(key in awareness for key in expected_keys), f"T1: Missing awareness keys. Got: {awareness.keys()}"
        test_results_summary.append("Test 1 (SNN w/ Mock Phi-3): Passed")
    except Exception as e: test_results_summary.append(f"Test 1: Failed - {type(e).__name__}: {e}"); print(traceback.format_exc())
    print(f"  Test 1 Result: {test_results_summary[-1]}")

    print("\n[Test 2: LLM/Phi-3 Fallback - SNN Disabled]")
    try:
        reset_manifold_for_test()
        with TempConfigOverride(ENABLE_SNN=False, ENABLE_LLM_API=True, VERBOSE_OUTPUT=False, LLM_PROVIDER="mock_for_snn_test", LLM_MODEL="phi-3-mock-fallback"): 
            thought_steps, response, awareness = think("Test SNN disabled fallback to Phi-3 mock", stream=False)
        assert "LLM monologue (SNN Bypass):" in response, "T2: Response should indicate SNN bypass."
        assert awareness.get("active_llm_fallback", False), "T2: active_llm_fallback should be True."
        assert awareness.get("primary_concept_coord") is not None, "T2: Primary coord missing in fallback."
        assert all(key in awareness for key in expected_keys), f"T2: Missing awareness keys. Got: {awareness.keys()}"
        test_results_summary.append("Test 2 (SNN Disabled Fallback): Passed")
    except Exception as e: test_results_summary.append(f"Test 2: Failed - {type(e).__name__}: {e}"); print(traceback.format_exc())
    print(f"  Test 2 Result: {test_results_summary[-1]}")
    
    print("\n[Test 3: Actual Phi-3 API Call (Conceptual - May Fallback to Mock)]")
    try:
        reset_manifold_for_test()
        with TempConfigOverride(ENABLE_SNN=False, ENABLE_LLM_API=True, VERBOSE_OUTPUT=True): 
            print("  (Test 3 Note: Check verbose output for API call attempt or mock usage. This test assumes LLM config in config.py points to a testable endpoint or mock.)")
            thought_steps, response, awareness = think("Explain quantum entanglement briefly", stream=False)
        assert len(response) > 0, "T3: Response is empty."
        assert awareness.get("primary_concept_coord") is not None, "T3: Primary coord missing."
        assert all(key in awareness for key in expected_keys), f"T3: Missing awareness keys. Got: {awareness.keys()}"
        # A more robust check would be to see if "mock_phi3_data_used" event was logged during this test if mock was used.
        # For simplicity, checking thought_steps as done previously.
        mock_used_indicator = any("mock_phi3_data_used" in str(step) or "Using mock Phi-3 data" in str(step) for step in thought_steps if isinstance(step, str))
        if mock_used_indicator: 
             print("  Test 3 Observation: Actual Phi-3 API call fell back to mock data as expected (no live server or specific mock provider for this test).")
        else:
             print("  Test 3 Observation: Actual Phi-3 API call may have been attempted (check logs for success/failure).")
        test_results_summary.append("Test 3 (Phi-3 API Call Conceptual): Passed (check logs)")
    except Exception as e: test_results_summary.append(f"Test 3: Failed - {type(e).__name__}: {e}"); print(traceback.format_exc())
    print(f"  Test 3 Result: {test_results_summary[-1]}")
    
    print("\n--- brain.py (Enhanced) Test Summary ---")
    all_passed = all("Passed" in res for res in test_results_summary)
    for res_str in test_results_summary: print(f"  {res_str}")
    if all_passed: print("\nSUCCESS: All brain.py (Enhanced) self-tests passed (or fell back to mock gracefully)!")
    else: print("\nFAILURE: One or more brain.py (Enhanced) self-tests failed.", file=sys.stderr)
    print("--- Testing brain.py (Enhanced) complete ---")

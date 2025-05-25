# Sophia_Alpha2/core/gui.py
"""
Streamlit-based Graphical User Interface for Sophia_Alpha2.
Allows interaction with the dialogue system and visualization of persona awareness.
"""
import streamlit as st
import sys
import os
import time # For simulating streaming if needed
import json # For pretty printing dicts/lists if needed

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
    from . import dialogue 
    from . import persona  
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_for_core = os.path.abspath(os.path.join(current_dir, "..")) 
    if project_root_for_core not in sys.path:
        sys.path.insert(0, project_root_for_core)
    try:
        from core import dialogue, persona
    except ModuleNotFoundError as e:
        st.error(f"GUI Error: Failed to import core modules. Ensure 'core' is in PYTHONPATH or accessible. Details: {e}")
        class MockModule: pass
        dialogue = MockModule()
        dialogue.generate_response = lambda x, stream=False: (f"ERROR: Core 'dialogue' module not loaded. Input: {x}", ["Dialogue module error"], {"error": "dialogue module not loaded"})
        dialogue.get_dialogue_persona = lambda: MockModule()
        mock_persona_instance = MockModule()
        mock_persona_instance.awareness = {"error": "Persona module not loaded"}
        mock_persona_instance.get_intro = lambda: "Persona module not loaded"
        dialogue.get_dialogue_persona = lambda: mock_persona_instance
        
        if 'config' not in globals() or not hasattr(config, 'PERSONA_NAME'):
            config = MockModule() 
            config.PERSONA_NAME = "Sophia (Error Mode)"
            config.VERBOSE_OUTPUT = True


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title=f"{getattr(config, 'PERSONA_NAME', 'Sophia')} - Cognitive Interface",
    page_icon="🧠", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# --- Application State Management (using Streamlit session state) ---
if 'dialogue_history' not in st.session_state:
    st.session_state.dialogue_history = [] 
if 'persona_instance' not in st.session_state:
    try:
        st.session_state.persona_instance = dialogue.get_dialogue_persona()
    except Exception as e_init_persona: 
        st.error(f"Failed to initialize Persona: {e_init_persona}")
        st.session_state.persona_instance = None 
if 'stream_thoughts_gui' not in st.session_state: # Renamed from stream_thoughts to avoid conflict
    st.session_state.stream_thoughts_gui = getattr(config, 'VERBOSE_OUTPUT', False) 
if 'last_thought_steps' not in st.session_state:
    st.session_state.last_thought_steps = []
if 'last_awareness_metrics' not in st.session_state:
    st.session_state.last_awareness_metrics = {}


# --- Helper Functions ---
def add_to_dialogue_history(speaker: str, message: str):
    st.session_state.dialogue_history.append({"speaker": speaker, "message": message})

def get_persona_name():
    if st.session_state.persona_instance and hasattr(st.session_state.persona_instance, 'name'):
        return st.session_state.persona_instance.name
    return getattr(config, 'PERSONA_NAME', 'Sophia')


# --- Sidebar (col2 equivalent) for Controls and Persona Awareness ---
with st.sidebar:
    st.title("Controls & Awareness")
    st.divider()

    # Overall Persona State (refreshes on rerun, sourced from the persona instance)
    st.subheader("Persona State")
    if st.session_state.persona_instance and hasattr(st.session_state.persona_instance, 'get_intro'):
        st.markdown(f"**{st.session_state.persona_instance.name}** ({getattr(st.session_state.persona_instance, 'mode', 'N/A')})")
        if hasattr(st.session_state.persona_instance, 'traits'):
            st.caption("Core Traits: " + ", ".join(st.session_state.persona_instance.traits[:3]) + "...")
        # Display overall awareness from persona (which is updated after each interaction)
        # This serves as the "current" state of the persona.
        st.markdown("**Overall Awareness:**")
        awareness_data_overall = st.session_state.persona_instance.awareness
        for key, value in awareness_data_overall.items():
            display_key = key.replace('_', ' ').title()
            if isinstance(value, float): 
                st.text(f"  {display_key}: {value:.3f}") 
            elif isinstance(value, (list, tuple)): 
                coord_str = ", ".join([f"{v:.2f}" if isinstance(v, float) else str(v) for v in value])
                st.text(f"  {display_key}: ({coord_str})")
            else: 
                st.text(f"  {display_key}: {value}")
    else:
        st.markdown("**Persona:** Not available.")
    st.divider()

    # Awareness Metrics Dashboard (from the *last specific interaction*)
    st.subheader("Last Interaction Metrics")
    if st.session_state.last_awareness_metrics:
        metrics = st.session_state.last_awareness_metrics
        st.metric("Curiosity", f"{metrics.get('curiosity', 0):.2f}")
        st.metric("Context Stability", f"{metrics.get('context_stability', 0):.2f}")
        st.metric("Self-Evolution Rate", f"{metrics.get('self_evolution_rate', 0):.3f}")
        st.metric("Coherence", f"{metrics.get('coherence', 0):.2f}")
        
        fallback_status = "Yes" if metrics.get('active_llm_fallback') else "No"
        col1_metric, col2_metric = st.columns(2)
        with col1_metric:
            st.markdown(f"**LLM Fallback:**")
            st.markdown(f"> {fallback_status}")
        with col2_metric:
            primary_coord = metrics.get('primary_concept_coord')
            if primary_coord and isinstance(primary_coord, (list, tuple)) and len(primary_coord) == 4:
                 coord_str = f"({primary_coord[0]:.2f}, {primary_coord[1]:.2f}, {primary_coord[2]:.2f}, T:{primary_coord[3]:.2f})"
            else: coord_str = "N/A"
            st.markdown(f"**Focus Coord:**"); st.markdown(f"> {coord_str}")
    else:
        st.write("No interaction metrics yet from last exchange.")
    st.divider()
    
    st.subheader("Settings")
    st.session_state.stream_thoughts_gui = st.checkbox(
        "Show Thought Stream Expander", # Clarified label
        value=st.session_state.stream_thoughts_gui,
        help="If checked, detailed thought process logs from the brain (last interaction) will be shown below the response."
    )

    if st.button("Clear Dialogue History"):
        st.session_state.dialogue_history = []
        st.session_state.last_thought_steps = [] 
        st.session_state.last_awareness_metrics = {} 
        add_to_dialogue_history(get_persona_name(), "Dialogue cleared. How may I assist you further?")
        st.rerun() 

    if st.button("Reset Persona State"):
        if st.session_state.persona_instance and hasattr(st.session_state.persona_instance, '_initialize_default_state_and_save'):
            st.session_state.persona_instance._initialize_default_state_and_save()
            st.session_state.dialogue_history = []
            st.session_state.last_thought_steps = []
            st.session_state.last_awareness_metrics = {}
            add_to_dialogue_history(get_persona_name(), "Persona state has been reset to defaults. Dialogue cleared.")
            st.rerun()
        else:
            st.warning("Persona instance not available or does not support reset.")


# --- Main Chat Interface (col1 equivalent) ---
st.header(f"Chat with {get_persona_name()}")

# Display dialogue history
for chat_entry in st.session_state.dialogue_history:
    with st.chat_message(name=chat_entry["speaker"]): 
        st.markdown(chat_entry["message"])

# Thought Stream Expander - Displays thoughts from the last interaction if checkbox is ticked
if st.session_state.last_thought_steps and st.session_state.stream_thoughts_gui:
    with st.expander("Show Thought Stream (Last Interaction)", expanded=False):
        st.markdown("```text") 
        for step_idx, step_detail in enumerate(st.session_state.last_thought_steps):
            st.text(f"Step {step_idx+1}: {step_detail}") 
        st.markdown("```")

user_input = st.chat_input(f"What would you like to ask {get_persona_name()}?")

if user_input:
    add_to_dialogue_history("You", user_input)
    with st.chat_message(name="You"):
        st.markdown(user_input)

    with st.chat_message(name=get_persona_name()):
        with st.spinner("Thinking..."): 
            try:
                # Call the core dialogue logic, expecting 3 return values
                sophia_response_text, thought_steps, awareness_metrics = dialogue.generate_response(
                    user_input, 
                    stream_thought_steps=st.session_state.stream_thoughts_gui 
                )
                
                # Store results for display in this and next rerun
                st.session_state.last_thought_steps = thought_steps
                st.session_state.last_awareness_metrics = awareness_metrics
                
                response_placeholder = st.empty()
                full_response_streamed = ""
                for chunk in sophia_response_text.split(): 
                    full_response_streamed += chunk + " "
                    response_placeholder.markdown(full_response_streamed + "▌") 
                    time.sleep(0.03) 
                response_placeholder.markdown(full_response_streamed.strip()) 

                add_to_dialogue_history(get_persona_name(), sophia_response_text)

            except Exception as e:
                error_msg = f"Core System Error: Could not generate response. Details: {e}"
                st.error(error_msg)
                add_to_dialogue_history(get_persona_name(), f"SYSTEM_ERROR: {error_msg}")
                st.session_state.last_thought_steps = [f"Error: {e}"] 
                st.session_state.last_awareness_metrics = {"error": str(e)} 
                if getattr(config, 'VERBOSE_OUTPUT', False):
                    import traceback
                    st.text_area("Stack Trace:", traceback.format_exc(), height=200)
    st.rerun() # Rerun to immediately update the sidebar metrics and thought expander visibility


# --- Initial message if history is empty ---
if not st.session_state.dialogue_history: 
    if st.session_state.persona_instance and hasattr(st.session_state.persona_instance, 'get_intro'):
        add_to_dialogue_history(get_persona_name(), st.session_state.persona_instance.get_intro())
    else: 
        add_to_dialogue_history(get_persona_name(), "Welcome! I am currently initializing.")
    st.rerun() # Display initial message


def start_gui():
    if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT:
        print("GUI: `start_gui()` called. Streamlit app is running.")

if __name__ == "__main__":
    if hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT:
        print("GUI: Executed as __main__. Streamlit app should be running or will start if called by streamlit run.")
    # The initial message logic at the end of the script handles the first display.

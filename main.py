# Sophia_Alpha2/main.py
"""
Main entry point for the Sophia_Alpha2 Cognitive Architecture.
Handles command-line arguments for different interfaces (CLI, GUI)
and initializes the core components.
"""
import argparse
import sys
import os

# Step 3: Modify sys.path Handling
project_root_path = os.path.abspath(os.path.dirname(__file__))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

# Step 4: Update Imports
try:
    import config # Imports from the top-level 'config' package
    # Mitigator is not directly used in main.py; it's used within dialogue.py
    # from core.library import Mitigator # Removed as per Step 3 of current subtask
except ImportError as e:
    print(f"Fatal Error: Could not import 'config'. Exception: {e}", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    print("Ensure main.py is at project root and 'config' package is present.", file=sys.stderr)
    sys.exit(1)
except Exception as e_gen:
    print(f"Fatal Error: An unexpected error occurred during initial module imports: {e_gen}", file=sys.stderr)
    sys.exit(1)

# Core modules like brain, dialogue, gui will be imported conditionally/locally within main_logic.

def main_logic(cli_args): 
    """
    Main function to initialize and run Sophia based on command-line arguments.
    """
    # Apply verbose setting from cli_args to config.
    if cli_args.verbose: 
        if hasattr(config, 'VERBOSE_OUTPUT'): 
            if not config.VERBOSE_OUTPUT: 
                print("Main: Verbose output enabled by command-line argument (overriding config).")
            config.VERBOSE_OUTPUT = True
        else: 
            print("Main: config.VERBOSE_OUTPUT not set in config.py, enabling due to command-line argument.")
            config.VERBOSE_OUTPUT = True
    elif not hasattr(config, 'VERBOSE_OUTPUT'): 
        config.VERBOSE_OUTPUT = False


    if not hasattr(config, 'ENABLE_GUI'):
        if config.VERBOSE_OUTPUT:
            print("Main: config.ENABLE_GUI not set in config.py, defaulting to False.")
        config.ENABLE_GUI = False

    effective_interface = cli_args.interface
    if cli_args.interface == "gui" and not config.ENABLE_GUI:
        if config.VERBOSE_OUTPUT:
            print("Main: GUI interface requested via CLI, but disabled in config. Falling back to CLI.")
        effective_interface = "cli"
    
    if config.VERBOSE_OUTPUT:
        print("Main: Sophia Alpha2 - Cognitive Architecture Initializing...")
        print(f"Main: Effective Interface: {effective_interface}")
        if cli_args.query and effective_interface == "cli":
            print(f"Main: Single Query Mode Active. Query: '{cli_args.query}'")
        print(f"Main: Verbose Output (effective): {config.VERBOSE_OUTPUT}")

    # Ensure data/log directories
    try:
        if hasattr(config, 'ensure_path'):
            paths_to_ensure = [
                getattr(config, 'DATA_DIR', None), getattr(config, 'LOG_DIR', None),
                getattr(config, 'PERSONA_DIR', None), getattr(config, 'SYSTEM_LOG_PATH', None),
                getattr(config, 'PERSONA_PROFILE', None), getattr(config, 'ETHICS_DB_PATH', None),
                getattr(config, 'KNOWLEDGE_GRAPH_PATH', None),
                getattr(config, 'LIBRARY_LOG_PATH', getattr(config, 'MEMORY_LOG_PATH', None)) 
            ]
            for pth_val in paths_to_ensure:
                if pth_val:
                    is_likely_dir = (pth_val.endswith(os.sep) or 
                                   (not os.path.splitext(pth_val)[1] and (os.path.isdir(pth_val) or not os.path.exists(pth_val)) ) )
                    if is_likely_dir:
                         if not os.path.exists(pth_val):
                             os.makedirs(pth_val, exist_ok=True)
                             if config.VERBOSE_OUTPUT: print(f"Main: Created directory via os.makedirs: {pth_val}")
                         config.ensure_path(os.path.join(pth_val, ".ensure_dir_placeholder")) 
                    else: 
                         config.ensure_path(pth_val) 
        else: 
             if config.VERBOSE_OUTPUT: print("Main: config.ensure_path not found, attempting manual directory creation if needed.")
             for dir_attr_name in ['DATA_DIR', 'LOG_DIR', 'PERSONA_DIR']:
                dir_path = getattr(config, dir_attr_name, None)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    if config.VERBOSE_OUTPUT: print(f"Main: Manually created directory: {dir_path}")
    except Exception as e_dir:
        if config.VERBOSE_OUTPUT: print(f"Main: Warning - Error ensuring data/log directories: {e_dir}", file=sys.stderr)
    

    from core import dialogue # Import for all interface modes that use it

    if effective_interface == "gui":
        try:
            from core.gui import start_gui 
            if config.VERBOSE_OUTPUT: print("Main: Launching GUI interface...")
            start_gui() 
        except ImportError as e_imp: 
            print(f"GUI Error: Could not import or start GUI. {e_imp}", file=sys.stderr)
            print("Falling back to CLI mode. Please ensure Streamlit is installed ('pip install streamlit') and GUI components are correct if you intended to use the GUI.", file=sys.stderr)
            if config.VERBOSE_OUTPUT: print("Main: Starting CLI interface instead...")
            dialogue.dialogue_loop(enable_streaming_thoughts=config.VERBOSE_OUTPUT) 
        except Exception as e_gui: 
            print(f"An unexpected error occurred while trying to start the GUI: {e_gui}", file=sys.stderr)
            print("Falling back to CLI mode.")
            if config.VERBOSE_OUTPUT: 
                import traceback
                traceback.print_exc(file=sys.stderr)
                print("Main: Starting CLI interface instead...") 
            dialogue.dialogue_loop(enable_streaming_thoughts=config.VERBOSE_OUTPUT)

    elif effective_interface == "cli":
        # from core import brain # Removed, as dialogue.generate_response handles brain interaction
        
        if cli_args.query: 
            # Step 2: Refine Single-Query CLI Mode
            if config.VERBOSE_OUTPUT: print(f"Main: Processing single query: '{cli_args.query}'")
            
            # Use all returned values from generate_response
            thought_steps, response_text, awareness_metrics = dialogue.generate_response(
                cli_args.query, 
                stream_thought_steps=False # No console streaming from brain for single query output by default
            ) 
            
            # Get persona name from config for consistent display
            persona_name_for_display = config.PERSONA_NAME if hasattr(config, 'PERSONA_NAME') else "Sophia"
            print(f"\n{persona_name_for_display}'s Response:\n{response_text}")
            
            if config.VERBOSE_OUTPUT:
                if awareness_metrics: # Check if awareness_metrics is not None
                    print("\n--- Awareness Metrics (from this interaction) ---") 
                    for key, value in awareness_metrics.items():
                        if isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value) : 
                            formatted_value = ", ".join([f"{v:.3f}" for v in value]) 
                            print(f"  {key.replace('_', ' ').title()}: ({formatted_value})")
                        elif isinstance(value, float):
                            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
                        else:
                            print(f"  {key.replace('_', ' ').title()}: {value}")
                
                if thought_steps: # Optionally print thought steps
                    print("\n--- Thought Steps (from this interaction) ---")
                    for i, step in enumerate(thought_steps):
                        print(f"  Step {i+1}: {step}")
        else: 
            if config.VERBOSE_OUTPUT: print("Main: Starting CLI interface...") 
            dialogue.dialogue_loop(enable_streaming_thoughts=config.VERBOSE_OUTPUT)
    else: 
        print(f"Main Error: Unknown effective interface type '{effective_interface}'. This should not happen.", file=sys.stderr)
        sys.exit(1)

    if config.VERBOSE_OUTPUT:
        print("\nMain: Sophia Alpha2 session ended.")

if __name__ == "__main__":
    default_interface = "gui" if (hasattr(config, 'ENABLE_GUI') and config.ENABLE_GUI) else "cli"
    
    parser = argparse.ArgumentParser(
        description="Sophia Alpha2 Cognitive Architecture - Main Entry Point.",
        formatter_class=argparse.RawTextHelpFormatter 
    )
    parser.add_argument(
        "--interface",
        choices=["cli", "gui"],
        default=default_interface, 
        help=f"Specify the interface type (default: {default_interface}).\n"
             "  cli: Command-Line Interface.\n"
             "  gui: Graphical User Interface (Streamlit-based)."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Enter a single query for Sophia to process.\n"
             "If provided, CLI runs once for this query and exits.\n"
             "Ignored if --interface is 'gui'."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output for debugging and detailed logging.\n"
             "Overrides config.VERBOSE_OUTPUT if it's False or not set."
    )
    
    parsed_cli_args = parser.parse_args() 
    
    initial_verbose_flag = parsed_cli_args.verbose or (hasattr(config, 'VERBOSE_OUTPUT') and config.VERBOSE_OUTPUT)

    if initial_verbose_flag: 
        print("Main (`if __name__`): Initializing Sophia Alpha2...") 
        log_path_configured = hasattr(config, 'SYSTEM_LOG_PATH') and config.SYSTEM_LOG_PATH
        if log_path_configured:
            log_dir = os.path.dirname(config.SYSTEM_LOG_PATH)
            if (not log_dir and os.path.exists(project_root_path)) or \
               (log_dir and os.path.exists(log_dir)): 
                 print(f"Main (`if __name__`): System log path configured: {config.SYSTEM_LOG_PATH}")
            else: 
                 print(f"Main (`if __name__`): System log path directory ({log_dir if log_dir else 'current directory'}) for {config.SYSTEM_LOG_PATH} may not be present yet (expected to be created by main_logic or config's ensure_path).")
        elif initial_verbose_flag : 
             print("Main (`if __name__`): System log path not configured in config.py.")
    
    main_logic(parsed_cli_args)

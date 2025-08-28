import json
import os
from pathlib import Path

def get_all_actions_and_risk():
    """
    Load the two specific files from traces_with_risk directory.
    
    Returns:
        tuple: (detailed_graph_dict, reactflow_graph_dict)
    """
    traces_dir = Path("jailbreak_eval/data/traces_with_risk")
    
    if not traces_dir.exists():
        print(f"Directory {traces_dir} does not exist")
        return None, None
    
    # Specific file names from the directory
    detailed_graph_file = "detailed_graph_langgraph_multi_trace.json"
    reactflow_graph_file = "reactflow_graph_with_multi_trace_with_risk.json"
    
    detailed_graph_dict = None
    reactflow_graph_dict = None
    
    # Load detailed graph file
    try:
        detailed_path = traces_dir / detailed_graph_file
        with open(detailed_path, 'r', encoding='utf-8') as f:
            detailed_graph_dict = json.load(f)
        print(f"Loaded {detailed_graph_file}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {detailed_graph_file}: {e}")
    except Exception as e:
        print(f"Error reading {detailed_graph_file}: {e}")
    
    # Load reactflow graph file
    try:
        reactflow_path = traces_dir / reactflow_graph_file
        with open(reactflow_path, 'r', encoding='utf-8') as f:
            reactflow_graph_dict = json.load(f)
        print(f"Loaded {reactflow_graph_file}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {reactflow_graph_file}: {e}")
    except Exception as e:
        print(f"Error reading {reactflow_graph_file}: {e}")
    
    return detailed_graph_dict, reactflow_graph_dict

if __name__ == "__main__":
    detailed_graph_dict, reactflow_graph_dict = get_all_actions_and_risk()
    
    if detailed_graph_dict is not None:
        print(f"Detailed graph loaded successfully with {len(detailed_graph_dict)} keys")
    else:
        print("Failed to load detailed graph")
        
    if reactflow_graph_dict is not None:
        print(f"Reactflow graph loaded successfully with {len(reactflow_graph_dict)} keys")
    else:
        print("Failed to load reactflow graph")
    
    # Extract actions from detailed graph and add risk data
    if detailed_graph_dict is not None and "actions" in detailed_graph_dict and reactflow_graph_dict is not None:
        all_actions = []
        
        # Create a mapping of action IDs to risk values from reactflow graph
        risk_mapping = {}
        if "action" in reactflow_graph_dict and "nodes" in reactflow_graph_dict["action"]:
            for node in reactflow_graph_dict["action"]["nodes"]:
                if node["id"].startswith("action_") and "average_jailbreak_ASR" in node["data"]:
                    risk_mapping[node["data"]["label"]] = node["data"]["average_jailbreak_ASR"]
        
        print(f"Risk mapping created with {len(risk_mapping)} entries")
        print(f"Number of action lists (traces): {len(detailed_graph_dict['actions'])}")
        
        # Iterate through each action list (each list starts with human_input)
        for i, action_list in enumerate(detailed_graph_dict["actions"]):
            actions_in_this_list = len(action_list) - 1  # Subtract 1 for human_input
            print(f"  Trace {i}: {actions_in_this_list} actions (excluding human_input)")
            
            # Skip the first item (human_input) and add the rest to all_actions
            for action in action_list[1:]:  # Skip index 0 (human_input)
                # Add risk value to the action
                action_label = action.get("label", "")
                if action_label in risk_mapping:
                    action["risk"] = risk_mapping[action_label]
                    print(f"    Added risk {risk_mapping[action_label]:.3f} to {action_label}")
                else:
                    action["risk"] = None
                    print(f"    No risk data found for {action_label}")
                
                all_actions.append(action)
        
        print(f"Total actions extracted (excluding human_input): {len(all_actions)}")
        
        # Count how many actions have risk data
        actions_with_risk_count = sum(1 for action in all_actions if action.get("risk") is not None)
        print(f"Actions with risk data: {actions_with_risk_count}/{len(all_actions)}")
        
        # Save actions with risk to JSON file
        output_file = Path("jailbreak_eval/data/traces_with_risk/actions_with_risk.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_actions, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved {len(all_actions)} actions to {output_file}")
        except Exception as e:
            print(f"Error saving file {output_file}: {e}")
        
    else:
        print("Missing data: detailed graph or reactflow graph not available")

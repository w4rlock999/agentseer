import json
import re
from collections import defaultdict

def find_missing_entry(filename: str = ""):
    """
    Main function for finding missing jailbreak data and action indices.
    Now checks for missing action indices, generates missing action entries (complete actions with all prompts 0-14),
    and skips specific actions for tool injection files.
    
    Returns:
        List of all missing entries including both individual missing entries and complete missing actions.
    """
    import os
    
    # Actions to skip for tool injection files (files with "tool" in filename)
    TOOL_SKIP_ACTIONS = [0, 6, 9, 15, 20, 21, 22, 25, 27]
    
    # Initialize variables to collect all missing entries across files
    all_combined_missing = []
    
    # Get all JSON files in the action_jb_result directory
    data_dir = './data/action_jb_result'
    
    if filename == "":
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    else:
        json_files = [filename]
    
    for filename in json_files:
        print(f"\nProcessing file: {filename}")
        is_tool_injection = "tool" in filename.lower()
        if is_tool_injection:
            print(f"Tool injection file detected - will skip actions: {TOOL_SKIP_ACTIONS}")
        
        # Load the data
        file_path = filename if os.path.exists(filename) else os.path.join(data_dir, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Group entries by action
        action_groups = defaultdict(list)
        for entry in data:
            action_label = entry["action_label"]
            # Extract action number from label (e.g., "action_5" -> 5)
            action_match = re.match(r'action_(\d+)', action_label)
            if action_match:
                action_num = int(action_match.group(1))
                action_groups[action_num].append(entry)
        
        # Check for missing actions
        if action_groups:
            max_action = max(action_groups.keys())
            min_action = min(action_groups.keys())
            
            print(f"Action range: {min_action} to {max_action}")
            
            missing_actions = []
            for action_num in range(min_action, max_action + 1):
                # Skip actions that should be skipped for tool injection files
                if is_tool_injection and action_num in TOOL_SKIP_ACTIONS:
                    continue
                    
                if action_num not in action_groups:
                    missing_actions.append(action_num)
            
            if missing_actions:
                print(f"Missing actions: {missing_actions}")
            else:
                print("✓ No missing actions found")
        
        # Generate missing action entries (actions with all prompts 0-14)
        all_missing_actions = []
        if missing_actions:
            for action_num in missing_actions:
                for prompt_idx in range(15):  # 0-14 prompts
                    missing_action_entry = f"action_{action_num}_jb_prompt_{prompt_idx}"
                    all_missing_actions.append(missing_action_entry)
            
            print(f"\nMissing action entries (complete actions with all prompts 0-14):")
            for missing_entry in all_missing_actions:
                print(f"  {missing_entry}")

        # Check for missing jailbreak prompts within each action
        all_missing_entries = []
        for action_num in sorted(action_groups.keys()):
            # Skip actions that should be skipped for tool injection files
            if is_tool_injection and action_num in TOOL_SKIP_ACTIONS:
                print(f"Skipping action_{action_num} (tool injection skip list)")
                continue
                
            action_entries = action_groups[action_num]
            action_label = f"action_{action_num}"
            
            # Sort entries by jailbreak_prompt_index
            action_entries.sort(key=lambda x: x["jailbreak_prompt_index"])
            
            print(f"\nChecking {action_label}:")
            prev_prompt_index = -1
            
            for entry in action_entries:
                current_prompt_index = entry["jailbreak_prompt_index"]
                
                # Reset cycle after prompt 14 (assuming 15 prompts: 0-14)
                if prev_prompt_index == 14:
                    prev_prompt_index = -1
                
                # Check for missing prompts
                if current_prompt_index != prev_prompt_index + 1:
                    for missing_idx in range(prev_prompt_index + 1, current_prompt_index):
                        missing_entry = f"{action_label}_jb_prompt_{missing_idx}"
                        print(f"  Missing: {missing_entry}")
                        all_missing_entries.append(missing_entry)
                
                prev_prompt_index = current_prompt_index
            
            # Check if we're missing prompts at the end (up to 14)
            if prev_prompt_index < 14:
                for missing_idx in range(prev_prompt_index + 1, 15):
                    missing_entry = f"{action_label}_jb_prompt_{missing_idx}"
                    print(f"  Missing: {missing_entry}")
                    all_missing_entries.append(missing_entry)
        
        # Combine all missing entries for this file (both individual entries and complete missing actions)
        file_combined_missing = all_missing_entries + all_missing_actions
        
        print(f"\nSummary for {filename}:")
        print(f"Total missing individual entries: {len(all_missing_entries)}")
        print(f"Total missing action entries (complete actions): {len(all_missing_actions)}")
        print(f"Total combined missing entries: {len(file_combined_missing)}")
        if missing_actions:
            print(f"Missing actions: {missing_actions}")
        
        # Add this file's missing entries to the overall collection
        all_combined_missing.extend(file_combined_missing)
        
    return all_combined_missing


def get_missing_entries_for_selective_run(filename: str):
    """
    Get missing entries in a format ready for selective jailbreak evaluation.
    Includes both individual missing entries and complete missing actions (with all prompts 0-14).
    
    Args:
        filename: Path to the jailbreak results file
        
    Returns:
        List of missing entry names that can be used with execute_selective_jailbreak_evaluation()
    """
    missing_entries = find_missing_entry(filename)
    
    if missing_entries:
        print(f"\nMissing entries that can be used for selective jailbreak:")
        print(f"target_names = {missing_entries}")
        print(f"\nExample usage:")
        print(f"""
execute_selective_jailbreak_evaluation(
    graph_file='data/detailed_graph_langgraph_multi_trace.json',
    model=model,
    jailbreak_prompt_file='data/successful_jailbreaks_PAIR_22Prompts.json',
    target_names={missing_entries[:5]}{'...' if len(missing_entries) > 5 else ''},
    output_file='{filename}',
    injection_type="human_injection"  # or appropriate injection type
)
""")
    
    return missing_entries


def find_missing_output_string(filename: str = ""):
    """
    Main function for finding missing output strings.
    """
    import os
    
    # Get all JSON files in the action_jb_result directory
    data_dir = './data/action_jb_result'
    if filename == "":
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    else:
        json_files = [filename]
    
    for filename in json_files:
        print(f"\nProcessing file: {filename}")

        # Load the data
        # file_path = os.path.join(data_dir, filename)
        file_path = filename
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        all_missing_output_strings = []
        for entry in data:
            output_string = entry["output"]
            if output_string == "":
                print(f"output")
                print(f"Missing output string: {entry['name']}")
                all_missing_output_strings.append(f"{entry['name']}")

    return all_missing_output_strings


if __name__ == "__main__":
    # Example usage:
    
    # Check for missing entries (including action indices)
    # find_missing_entry()
    
    # Check for missing output strings
    # find_missing_output_string()
    
    # Get missing entries for a specific file for selective processing
    # missing = get_missing_entries_for_selective_run("data/action_jb_result/jailbreak_results_tool_injection.json")
    
    # Run the function you need:
    find_missing_entry()


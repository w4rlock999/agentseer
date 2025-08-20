import json
import os

def process_file(file_path):
    """Process a single JSON file."""
    print(f"Processing file: {file_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process the data
    cur_index = 0
    cur_action_index = 0
    for item in data:
        action_label = item["action_label"]
        action_index = int(action_label.split("_")[1])
        
        if not cur_action_index == action_index:
            cur_action_index = action_index
            cur_index+=1
        
        if not (action_index == cur_index):

            # Modify the action label in the item
            corrected_action_label = f"action_{cur_index}"
            item["action_label"] = corrected_action_label
            
            # Also modify the name field if it exists and follows the pattern
            if "name" in item and item["name"] and item["name"].startswith(f"action_{action_index}_"):
                # Extract the suffix after the action index
                name_parts = item["name"].split("_", 2)  # Split into ["action", "index", "suffix"]
                if len(name_parts) >= 3:
                    suffix = name_parts[2]
                    item["name"] = f"action_{cur_index}_{suffix}"
                else:
                    # If no suffix, just update the action number
                    item["name"] = f"action_{cur_index}"


            print("cur action label", action_label)
            print("action label should be action_",cur_index)

    # Write the modified data back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ File saved: {file_path}")
    return True

def process_tool_file(file_path):
    """Process a single JSON file."""
    print(f"Processing file: {file_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process the data
    cur_index = 0
    cur_action_index = 0
    prev_jailbreak_prompt_index = -1
    for item in data:
        action_label = item["action_label"]
        action_index = int(action_label.split("_")[1])
        jailbreak_prompt_index = item.get("jailbreak_prompt_index", 0)
        
        # Increment when either action changes OR jailbreak_prompt_index resets to 0
        if (not cur_action_index == action_index) or (jailbreak_prompt_index == 0 and prev_jailbreak_prompt_index > 0):
            cur_action_index = action_index
            cur_index+=1
            print(f"DEBUG: New action sequence detected (action: {action_index}, prompt_idx: {jailbreak_prompt_index}), cur_index incremented to {cur_index}")
            while cur_index in [0, 6, 9, 15, 20, 21, 22, 25, 27]:
                print(f"DEBUG: Skipping cur_index {cur_index}")
                cur_index+=1
            print(f"DEBUG: Final cur_index for action {action_index}: {cur_index}")
        
        prev_jailbreak_prompt_index = jailbreak_prompt_index
        
        if not (action_index == cur_index):
            
            # Modify the action label in the item
            corrected_action_label = f"action_{cur_index}"
            item["action_label"] = corrected_action_label
            
            # Also modify the name field if it exists and follows the pattern
            if "name" in item and item["name"] and item["name"].startswith(f"action_{action_index}_"):
                # Extract the suffix after the action index
                name_parts = item["name"].split("_", 2)  # Split into ["action", "index", "suffix"]
                if len(name_parts) >= 3:
                    suffix = name_parts[2]
                    item["name"] = f"action_{cur_index}_{suffix}"
                else:
                    # If no suffix, just update the action number
                    item["name"] = f"action_{cur_index}"

            print("cur action label", action_label)
            print(f"corrected action label to action_{cur_index}")

    # Write the modified data back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ File saved: {file_path}")
    return True
         

def main():
    """Main function to process all files in the action_jb_result directory."""
    result_dir = "data/action_jb_result"
    
    if not os.path.exists(result_dir):
        print(f"❌ Directory not found: {result_dir}")
        return
    
    # Find all JSON files in the directory
    json_files = [f for f in os.listdir(result_dir) if f.endswith('.json') and not f.endswith('.backup')]
    
    if not json_files:
        print(f"❌ No JSON files found in {result_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for f in json_files:
        print(f"  - {f}")
    
    # Process each file
    modified_count = 0
    for filename in json_files:
        file_path = os.path.join(result_dir, filename)

        if 'tool' in filename:
            if process_tool_file(file_path):
                modified_count += 1
        else:
            if process_file(file_path):
                modified_count += 1
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {modified_count}/{len(json_files)} files were modified")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

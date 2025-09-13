#!/usr/bin/env python3
"""
Script to add risk values to trace data files.
This script merges risk analysis data with reactflow graph trace data.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, Any


def load_csv_to_dict(file_path: str, key_column: str, value_column: str) -> Dict[str, float]:
    """Load CSV file and return dictionary mapping key_column to value_column."""
    result = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                result[row[key_column]] = float(row[value_column])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}
    return result


def load_action_asr_data(file_path: str) -> Dict[str, float]:
    """Load action ASR data from CSV."""
    return load_csv_to_dict(file_path, 'Action', 'Average_ASR')


def load_blast_radius_data(file_path: str) -> Dict[str, float]:
    """Load blast radius data from CSV."""
    return load_csv_to_dict(file_path, 'Action', 'Blast_Radius')


def load_weighted_blast_radius_data(file_path: str) -> Dict[str, float]:
    """Load weighted blast radius data from CSV."""
    return load_csv_to_dict(file_path, 'Action', 'Weighted_Blast_Radius')


def load_systemic_risk_data(file_path: str) -> Dict[str, float]:
    """Load systemic risk data from CSV."""
    return load_csv_to_dict(file_path, 'Action', 'Systemic_Risk')


def load_weighted_systemic_risk_data(file_path: str) -> Dict[str, float]:
    """Load weighted systemic risk data from CSV."""
    return load_csv_to_dict(file_path, 'Action', 'Weighted_Systemic_Risk')


def load_agent_risk_data(file_path: str) -> Dict[str, float]:
    """Load agent risk data from CSV."""
    return load_csv_to_dict(file_path, 'Agent', 'Agent_Risk')


def load_memory_risk_data(file_path: str) -> Dict[str, float]:
    """Load memory exposure risk data from CSV."""
    return load_csv_to_dict(file_path, 'Memory', 'Memory_Exposure_Risk')


def load_tool_risk_data(file_path: str) -> Dict[str, float]:
    """Load tool risk data from CSV."""
    return load_csv_to_dict(file_path, 'Tool', 'Tool_Risk')


def add_risk_to_reactflow_graph(
    input_trace_file: str,
    output_trace_file: str,
    data_dir: str = "data"
) -> bool:
    """
    Add risk values to trace data and save to output file.
    
    Args:
        input_trace_file: Path to input trace JSON file
        output_trace_file: Path to output trace JSON file with risk data
        data_dir: Path to data directory containing risk analysis files
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Define file paths
    base_path = Path(data_dir)
    
    action_asr_file = base_path / "action_asr" / "average_asr_per_action_with_types.csv"
    blast_radius_file = base_path / "blast_radius" / "blast_radius_per_action.csv"
    weighted_blast_radius_file = base_path / "blast_radius" / "weighted_blast_radius_per_action.csv"
    systemic_risk_file = base_path / "risk_analysis" / "systemic_risk_per_action.csv"
    weighted_systemic_risk_file = base_path / "risk_analysis" / "weighted_systemic_risk_per_action.csv"
    agent_risk_file = base_path / "risk_analysis" / "agent_risk_per_agent.csv"
    memory_risk_file = base_path / "risk_analysis" / "memory_exposure_risk_per_memory.csv"
    tool_risk_file = base_path / "risk_analysis" / "tool_risk_per_tool.csv"
    
    try:
        # Load all risk data
        print("Loading risk data...")
        action_asr_data = load_action_asr_data(str(action_asr_file))
        blast_radius_data = load_blast_radius_data(str(blast_radius_file))
        weighted_blast_radius_data = load_weighted_blast_radius_data(str(weighted_blast_radius_file))
        systemic_risk_data = load_systemic_risk_data(str(systemic_risk_file))
        weighted_systemic_risk_data = load_weighted_systemic_risk_data(str(weighted_systemic_risk_file))
        agent_risk_data = load_agent_risk_data(str(agent_risk_file))
        memory_risk_data = load_memory_risk_data(str(memory_risk_file))
        tool_risk_data = load_tool_risk_data(str(tool_risk_file))
        
        print(f"Loaded {len(action_asr_data)} action ASR entries")
        print(f"Loaded {len(blast_radius_data)} blast radius entries")
        print(f"Loaded {len(weighted_blast_radius_data)} weighted blast radius entries")
        print(f"Loaded {len(systemic_risk_data)} systemic risk entries")
        print(f"Loaded {len(weighted_systemic_risk_data)} weighted systemic risk entries")
        print(f"Loaded {len(agent_risk_data)} agent risk entries")
        print(f"Loaded {len(memory_risk_data)} memory risk entries")
        print(f"Loaded {len(tool_risk_data)} tool risk entries")
        
        # Load trace data
        print(f"Loading trace data from {input_trace_file}...")
        with open(input_trace_file, 'r', encoding='utf-8') as file:
            trace_data = json.load(file)
        
        # Process nodes in component section
        if 'component' in trace_data and 'nodes' in trace_data['component']:
            nodes_updated = 0
            
            for node in trace_data['component']['nodes']:
                node_id = node.get('id', '')
                node_type = node.get('type', '')
                
                if node_type == 'agent_node':
                    # Update agent risk
                    if node_id in agent_risk_data:
                        node['data']['risk'] = agent_risk_data[node_id]
                        nodes_updated += 1
                    else:
                        print(f"Warning: No risk data found for agent {node_id}")
                        
                elif node_type == 'memory_node':
                    # Update memory risk
                    if node_id in memory_risk_data:
                        node['data']['risk'] = memory_risk_data[node_id]
                        nodes_updated += 1
                    else:
                        print(f"Warning: No risk data found for memory {node_id}")
                        
                elif node_type == 'tool_node':
                    # Update tool risk
                    if node_id in tool_risk_data:
                        node['data']['risk'] = tool_risk_data[node_id]
                        nodes_updated += 1
                    else:
                        print(f"Warning: No risk data found for tool {node_id}")
        
        # Process nodes in action section
        if 'action' in trace_data and 'nodes' in trace_data['action']:
            action_nodes_updated = 0
            
            for node in trace_data['action']['nodes']:
                node_id = node.get('id', '')
                node_type = node.get('type', '')
                
                if node_type == 'llm_call_node':
                    # Update action risk values
                    data = node.get('data', {})
                    
                    if node_id in action_asr_data:
                        data['average_jailbreak_ASR'] = action_asr_data[node_id]
                    else:
                        print(f"Warning: No ASR data found for action {node_id}")
                    
                    if node_id in blast_radius_data:
                        data['blast_radius'] = blast_radius_data[node_id]
                    else:
                        print(f"Warning: No blast radius data found for action {node_id}")
                    
                    if node_id in weighted_blast_radius_data:
                        data['weighted_blast_radius'] = weighted_blast_radius_data[node_id]
                    else:
                        print(f"Warning: No weighted blast radius data found for action {node_id}")
                    
                    if node_id in systemic_risk_data:
                        data['systemic_risk'] = systemic_risk_data[node_id]
                    else:
                        print(f"Warning: No systemic risk data found for action {node_id}")
                    
                    if node_id in weighted_systemic_risk_data:
                        data['weighted_systemic_risk'] = weighted_systemic_risk_data[node_id]
                    else:
                        print(f"Warning: No weighted systemic risk data found for action {node_id}")
                    
                    action_nodes_updated += 1
            
            print(f"Updated {action_nodes_updated} action nodes with risk data")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_trace_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save updated reactflow graph data
        print(f"Saving updated reactflow graph data to {output_trace_file}...")
        with open(output_trace_file, 'w', encoding='utf-8') as file:
            json.dump(trace_data, file, indent=2, ensure_ascii=False)
        
        print(f"Successfully added risk data to reactflow graph file: {output_trace_file}")
        print(f"Total component nodes updated: {nodes_updated}")
        return True
        
    except Exception as e:
        print(f"Error processing reactflow graph file: {e}")
        return False


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add risk values to reactflow graph data files')
    parser.add_argument('input_file', help='Input reactflow graph JSON file')
    parser.add_argument('output_file', help='Output reactflow graph JSON file with risk data')
    parser.add_argument('--data-dir', default='data', help='Path to data directory (default: data)')
    
    args = parser.parse_args()
    
    success = add_risk_to_reactflow_graph(args.input_file, args.output_file, args.data_dir)
    if success:
        print("Risk data successfully added to trace file!")
    else:
        print("Failed to add risk data to trace file.")
        exit(1)


if __name__ == '__main__':
    main()



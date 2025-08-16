from flask import Flask, request, jsonify
import json
import re, ast
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict

app = Flask(__name__)

@dataclass
class Component:
    component_type: str
    component_name: str
    component_description: str

@dataclass
class Node:
    id: str
    input: str
    output: str
    # interaction_components: List[int]

class TraceGraph:

    def __init__(self, trace_file: str):
        with open(trace_file, 'r') as f:
            self.trace_data = json.load(f)
        self.components = []
        self.nodes = []
        self.agents = []
        self.tools = []
        self.long_term_memory = []
        self.component_map = {}  # Maps component names to indices
        self.basic_graph = {}
        
    def parse_tool(self):
        """Extract tool calls from the trace data"""

    def get_kickoff_span(self):
        """Extract unique components from the trace data"""
        
        for span in self.trace_data['data']['spans']:
            span_name = span['name']
            if span_name == "Crew.kickoff":
                return span
            
    def get_agent_spans(self):
        agent_execute_span = []

        for span in self.trace_data['data']['spans']:
            span_name = span['name']
            if span['attributes']["mlflow.spanType"] == "\"AGENT\"":
                agent_execute_span.append(span)
        
        return agent_execute_span
    
    def get_retriever_spans(self):
        retriever_span = []

        for span in self.trace_data['data']['spans']:
            span_name = span['name']
            if span['attributes']["mlflow.spanType"] == "\"RETRIEVER\"":
                retriever_span.append(span)
        
        return retriever_span

    def get_create_long_term_memory_spans(self):
        create_long_term_memory_spans = []
        retriever_spans = self.get_retriever_spans()


        for cur_retriever in retriever_spans:
            if "CrewAgentExecutor._create_long_term_memory_" in cur_retriever["name"]:
                create_long_term_memory_spans.append(cur_retriever)
        
        return create_long_term_memory_spans

    def extract_workflow_agents(self):
        kickoff_span = self.get_kickoff_span()
        agent_from_kickoff = kickoff_span["attributes"]["agents"]    

        agent_string = agent_from_kickoff.strip('"')

        # Regex to remove everything between 'tools': [ and ]
        # todo copy the string between 'tools': [ and ], and process as tools
        agent_string = re.sub(r"'tools':\s*\[.*?\]", "'tools': []", agent_string)
        agent_string = re.sub(r'\\"', '"', agent_string)

        agents_dict = ast.literal_eval(agent_string)
        for i, agent_obj in enumerate(agents_dict):
            # print(agent_obj)
            self.agents.append({
                "agent_id": f"{i}",
                "name": agent_obj["role"],
                "backstory": agent_obj["backstory"],
                "goal": agent_obj["goal"],
                "model": agent_obj["llm"]
            })
        return self.agents
    
    def extract_workflow_tools(self):
        
        agent_spans = self.get_agent_spans()

        for agent_span in agent_spans:
            cur_agent_name = agent_span["attributes"]["role"]
            tools_string = agent_span["attributes"]["tools"].strip('"')
            tools_string = re.sub(r'\\"', '"', tools_string)


            tool_dict = ast.literal_eval(tools_string)
            
            for tool in tool_dict:
                if not any(existing_tool["name"] == tool["function"]["name"] for existing_tool in self.tools):
                    self.tools.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"]
                    })

    def extract_workflow_memory(self):
        memory_creation_spans = self.get_create_long_term_memory_spans()

        for cur_memory_creation_span in memory_creation_spans:
            self.long_term_memory.append({
                "memory" : cur_memory_creation_span["attributes"]["mlflow.spanInputs"]
                # "task" : get_task_from_trace_id(trace_id = trace_id)
            })

    def extract_workflow_components(self):
        self.extract_workflow_agents()
        self.extract_workflow_tools()
        self.extract_workflow_memory()
        return self.agents, self.tools, self.long_term_memory
    
    def extract_nodes(self):
        """Extract nodes from LLM spans"""
        node_id = 1
        
        for span in self.trace_data['data']['spans']:
            if span['attributes'].get('mlflow.spanType', '').strip('"') == 'LLM':
                # Extract input and output
                inputs = json.loads(span['attributes'].get('mlflow.spanInputs', '{}'))
                outputs = span['attributes'].get('mlflow.spanOutputs', '')
                
                # Get parent span to determine interaction components
                parent_id = span['parent_id']
                parent_span = next((s for s in self.trace_data['data']['spans'] 
                                  if s['context']['span_id'] == parent_id), None)
                
                # linking current LLM call to components
                # interaction_components = []
                # if parent_span:
                #     parent_name = parent_span['name']
                #     if parent_name in self.component_map:
                #         interaction_components.append(self.component_map[parent_name])
                
                node = Node(
                    id=str(node_id),
                    input=json.dumps(inputs),
                    output=outputs,
                    # interaction_components=interaction_components
                )
                self.nodes.append(node)
                node_id += 1

        return self.nodes
    
    def generate_basic_graph(self):
        
        self.extract_workflow_components()
        self.extract_nodes()
        
        return {
            "components": {
                "agents": [
                    {
                        "name": a['name'],
                        "backstory": a['backstory'],
                        "goal": a['goal'],
                        "model": a['model']
                    } for a in self.agents
                ],
                "tools": [
                    {
                        "name": t['name'],
                        "description": t['description']
                    } for t in self.tools
                ],
                "memory": [
                    {
                        "value": m['memory']
                    } for m in self.long_term_memory
                ]
            },
            "nodes": [
                {
                    "id": n.id,
                    "input": n.input,
                    "output": n.output,
                    # "interaction_components": n.interaction_components
                }
                for n in self.nodes
            ]
        }
    

    def clean_text(self,text):
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def get_ngrams(self, text, n):
        # Split into tokens and generate n-grams
        tokens = text.split()
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def calculate_similarity_score(self, target, source):
        # Clean both strings
        target = self.clean_text(target)
        source = self.clean_text(source)
        
        # Generate n-grams for both strings
        target_ngrams = set()
        for n in range(1, 4):  # Use 1-3 grams
            target_ngrams.update(self.get_ngrams(target, n))
        
        source_ngrams = set()
        for n in range(1, 4):
            source_ngrams.update(self.get_ngrams(source, n))
        
        # Calculate similarity
        if not target_ngrams or not source_ngrams:
            return 0.0
        
        intersection = len(target_ngrams.intersection(source_ngrams))
        union = len(target_ngrams.union(source_ngrams))
        target_ngrams_len = len(target_ngrams)
        source_ngrams_len = len(source_ngrams)
        
        return intersection / target_ngrams_len if target_ngrams_len > 0 else 0.0

    def is_node_exec_by_agent(self, node, agent_name, agent_backstory, agent_goal):
        """
            Check if the node is executed by the agent
        """

        node_input = json.loads(node['input'])

        # Calculate individual scores
        name_score = self.calculate_similarity_score(agent_name, str(node_input))
        backstory_score = self.calculate_similarity_score(agent_backstory, str(node_input))
        goal_score = self.calculate_similarity_score(agent_goal, str(node_input))
        
        # Calculate average score
        avg_score = (name_score + backstory_score + goal_score) / 3
        
        # print(f"agent: {agent_name}, avg_score: {avg_score}")

        # Return True if average score is above threshold
        return avg_score > 0.9  # Adjust threshold as needed


    def is_use_tool(self, node_string : str, tool_name):

        node_string_clean = self.clean_text(node_string)

        tool_name_str = f"action {self.clean_text(tool_name)}"
        if tool_name_str in node_string_clean:
            return True
        return False
    
    def is_use_memory(self, node_string : str, memory_string : str):

        # Calculate individual scores
        memory_score = self.calculate_similarity_score(memory_string, node_string)
        
        # Return True if average score is above threshold
        return memory_score > 0.6  # Adjust threshold as needed

    def is_node_dependency(self, node, prev_node):
        """
            Check if the node is dependent on the previous node
        """

        cur_node_input = self.clean_text(str(json.loads(node['input'])))
        prev_node_output = self.clean_text(str(json.loads(prev_node['output'])))

        dependency_score = self.calculate_similarity_score(prev_node_output, cur_node_input)
        # print(f"dependency_score, {dependency_score}, prev_node_output: {prev_node_output}")
        return dependency_score > 0.8


    def generate_graph(self):
        """
            Generate the final graph structure
        """
        basic_graph = {}
        basic_graph = self.generate_basic_graph()

        # loop all node to get component relation
        for node_index, node in enumerate(basic_graph['nodes']):
                
            # Check each component to find matching agent
            # Initialize agent attributes
            node['agent_index'] = -1
            node['agent_name'] = ''
            node['tool_in_input'] = []
            node['tool_in_output'] = []
            node['memory_in_input'] = []
            node['memory_in_output'] = []
            node['dependency_nodes'] = []

            # Check each component to find matching agent
            for component_index, agent in enumerate(basic_graph['components']['agents']):
                    cur_agent_name = agent['name']
                    cur_agent_backstory = agent['backstory']
                    cur_agent_goal = agent['goal']

                    if self.is_node_exec_by_agent(node, cur_agent_name, cur_agent_backstory, cur_agent_goal):
                        node['agent_index'] = component_index
                        node['agent_name'] = cur_agent_name
                        break

            # Check each component to find matching tools
            for component_index, tool in enumerate(basic_graph['components']['tools']):
                cur_tool_name = tool['name']

                if self.is_use_tool(str(node['input']), cur_tool_name):
                    node['tool_in_input'].append(component_index)
                if self.is_use_tool(str(node['output']), cur_tool_name):
                    node['tool_in_output'].append(component_index)
                            
            # Check each component to find matching memory

            for component_index, memory in enumerate(basic_graph['components']['memory']):
                if self.is_use_memory(str(node['input']), memory['value']):
                    node['memory_in_input'].append(component_index)
                if self.is_use_memory(str(node['output']), memory['value']):
                    node['memory_in_output'].append(component_index)

            # Check each previous nodes to find matching dependency
            # print(f"cur_node_index: {node['id']}")
            # print(f"cur_node_input: {self.clean_text(str(json.loads(node['input'])))}")
            for cur_previous_node_index in range(node_index-1, -1, -1):

                prev_node = basic_graph['nodes'][cur_previous_node_index]
                # print(f"prev_node_index: {prev_node['id']}")

                # a longer dependency node check might be useful for propagation analysis later on
                if self.is_node_dependency(node, prev_node):
                    node['dependency_nodes'].append(prev_node['id'])
                else:
                    break  # Stop checking only when no dependency is found

                break


        # Initialize edges list
        basic_graph['edges'] = []
        
        # Generate edges for node dependency
        for i, cur_node in enumerate(basic_graph['nodes']):
            for dependency_node_id in cur_node['dependency_nodes']:
                # Check if the target node is already in the edges list
                edge = {
                    'source': dependency_node_id,
                    'target': cur_node['id'],
                }
                basic_graph['edges'].append(edge)

        # Generate edges for long term memory
        # Iterate through nodes to find memory connections
        for i, source_node in enumerate(basic_graph['nodes']):
            # Only look at nodes after current node
            if len(source_node['memory_in_output'])>0:
                for target_node in basic_graph['nodes'][i+1:]:
                    # Check if there's any memory connection between nodes
                    for memory_idx in source_node['memory_in_output']:
                        if memory_idx in target_node['memory_in_input']:
                            # Create edge from source to target
                            edge = {
                                'source': source_node['id'],
                                'target': target_node['id'],
                                'memory_index': memory_idx
                            }
                            basic_graph['edges'].append(edge)

        
        self.basic_graph = basic_graph
        
        return basic_graph
    
    def convert_graph_to_reactflow(self):
        """
            Convert the graph to reactflow format
        """

        graph = self.basic_graph

        # Convert nodes
        initial_nodes = [
            {
                "id": str(node["id"]),
                "position": {
                    "x": 150 if (len(node["memory_in_input"]) > 0 and i % 2 == 1) else (-150 if (len(node["memory_in_input"]) > 0 and i % 2 == 0) else 0),
                    "y": 100 * i
                },
                "data": {
                    "label": f"Node {node['id']}",
                    "agent_id": f"{node['agent_index']}",
                    "agent_name": f"{node['agent_name']}"
                },
                "type": "llm_call_node"  # Default type if not provided
            }
            for i, node in enumerate(graph['nodes'])
        ]

        # Convert edges
        initial_edges = [
            {
                "id": f"e{edge['source']}-{edge['target']}",
                "source": str(edge["source"]),
                "target": str(edge["target"]),
                "data": {
                    "from_memory": str("memory_index" in edge),
                    "memory_index": edge["memory_index"] if "memory_index" in edge else 'None'
                },
                "style": { "strokeDasharray": "5, 5" if "memory_index" in edge else 'none' }
            }
            for edge in graph['edges']
        ]

        # for edge in graph['edges']:
        #     print(edge['source'], edge['target'])

        return initial_nodes, initial_edges


@app.route('/process-text', methods=['POST'])
def process_text():
    try:
        # Parse incoming JSON
        data = request.get_json()
        text = data.get('text', '')

        print(f"got message: {text}")
        # Mocked processing
        response = {
            "message": f"Received text: {text}",
            "length": len(text)
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-graph', methods=['POST'])
def generate_graph():
    try:
        # Parse incoming JSON
        data = request.get_json()
        trace_file = data.get('trace_file', '')

        # Initialize TraceGraph
        trace_graph = TraceGraph(trace_file)
        
        # Generate the graph
        graph = trace_graph.generate_graph()
        
        # Convert to reactflow format
        nodes, edges = trace_graph.convert_graph_to_reactflow()

        return jsonify({
            "nodes": nodes,
            "edges": edges
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

import React, { useRef, useEffect, useState, useCallback } from 'react';
import './RightPanel.css';

interface RightPanelProps {
  selectedNode: any;
  width: number;
  setWidth: (w: number) => void;
}

interface Message {
  content: string;
  type: string;
  name?: string;
  tool_calls?: any[];
}

interface ActionInfo {
  id: string;
  input: Message[];
  output: {
    generations: Array<Array<{
      message: {
        content: string;
        additional_kwargs: {
          tool_calls?: any[];
        };
      };
    }>>;
  };
  agent_id: string;
  agent_name: string;
  input_components: string[];
  output_components: string[];
  average_jailbreak_ASR: number;
  blast_radius: number;
  weighted_blast_radius: number;
  systemic_risk: number;
  weighted_systemic_risk: number;
}

interface AgentInfo {
  name: string;
  backstory: string;
  goal: string;
  model: string;
  id: string;
  risk: number;
}

interface MemoryInfo {
  id: string;
  memory_content: string;
  memory_index: number;
  risk: number;
}

interface ToolInfo {
  tool_name: string;
  description: string;
  id: string;
  risk: number;
}

const MIN_WIDTH = 20; // Percentage
const MAX_WIDTH = 40; // Percentage

const RightPanel: React.FC<RightPanelProps> = ({ selectedNode, width, setWidth }) => {
  const panelRef = useRef<HTMLDivElement>(null);
  const [actionInfo, setActionInfo] = useState<ActionInfo | null>(null);
  const [agentInfo, setAgentInfo] = useState<AgentInfo | null>(null);
  const [memoryInfo, setMemoryInfo] = useState<MemoryInfo | null>(null);
  const [toolInfo, setToolInfo] = useState<ToolInfo | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [componentMap, setComponentMap] = useState<Record<string, any>>({});

  useEffect(() => {
    const loadInfo = async () => {
      if (selectedNode?.type === 'llm_call_node') {
        try {
          // Get graph structure from initial_flow.json
          const graphResponse = await fetch('/reactflow_graph_with_multi_trace.json');
          const graphData = await graphResponse.json();
          
          // Build component map from graph data
          const newComponentMap: Record<string, any> = {};
          try {
            if (graphData?.component?.nodes) {
              graphData.component.nodes.forEach((node: any) => {
                if (!node || !node.type || !node.data) return;
                
                if (node.type === 'agent_node' && node.data.agent_name) {
                  newComponentMap[node.id] = { type: 'agent', name: node.data.agent_name };
                } else if (node.type === 'memory_node' && node.data.memory_content) {
                  newComponentMap[node.id] = { 
                    type: 'memory', 
                    name: node.data.memory_content.substring(0, 30) + (node.data.memory_content.length > 30 ? '...' : '')
                  };
                } else if (node.type === 'tool_node' && node.data.tool_name) {
                  newComponentMap[node.id] = { type: 'tool', name: node.data.tool_name };
                }
              });
            }
          } catch (error) {
            console.warn('Error building component map:', error);
          }
          setComponentMap(newComponentMap);

          // Get action details from detailed_graph_multi_trace.json
          const detailsResponse = await fetch('/detailed_graph_multi_trace.json');
          const detailsData = await detailsResponse.json();
          
          try {
            // Find the action in the graph data for basic info
            const graphAction = graphData?.action?.nodes?.find((a: any) => a?.id === selectedNode?.id);
            
            if (graphAction?.data) {
              // Find detailed action data
              const detailedAction = detailsData?.actions?.flat()?.find((a: any) => a?.label === graphAction.data.label);
              
              setActionInfo({
                id: graphAction.id,
                input: detailedAction?.input || [],
                output: detailedAction?.output || { generations: [] },
                agent_id: graphAction.data.agent_id,
                agent_name: graphAction.data.agent_name,
                input_components: graphAction.data.input_components || [],
                output_components: graphAction.data.output_components || [],
                average_jailbreak_ASR: graphAction.data.average_jailbreak_ASR || 0,
                blast_radius: graphAction.data.blast_radius || 0,
                weighted_blast_radius: graphAction.data.weighted_blast_radius || 0,
                systemic_risk: graphAction.data.systemic_risk || 0,
                weighted_systemic_risk: graphAction.data.weighted_systemic_risk || 0
              });
            }
          } catch (error) {
            console.warn('Error processing action data:', error);
          }
          
          setAgentInfo(null);
          setMemoryInfo(null);
          setToolInfo(null);
        } catch (error) {
          console.error('Failed to load action info:', error);
          setActionInfo(null);
        }
      } else if (selectedNode?.type === 'agent_node') {
        try {
          const response = await fetch('/detailed_graph_multi_trace.json');
          const data = await response.json();
          const agent = data.component.nodes.find((a: any) => a.id === selectedNode.id && a.type === 'agent_node');
          if (agent) {
            setAgentInfo({
              id: agent.id,
              name: agent.data.agent_name,
              backstory: agent.data.backstory,
              goal: agent.data.goal,
              model: "gpt-4o-mini",
              risk: agent.data.risk || 0
            });
          }
          setActionInfo(null);
          setMemoryInfo(null);
          setToolInfo(null);
        } catch (error) {
          console.error('Failed to load agent info:', error);
          setAgentInfo(null);
        }
      } else if (selectedNode?.type === 'memory_node') {
        try {
          const response = await fetch('/detailed_graph_multi_trace.json');
          const data = await response.json();
          const memory = data.component.nodes.find((m: any) => m.id === selectedNode.id && m.type === 'memory_node');
          if (memory) {
            setMemoryInfo({
              id: memory.id,
              memory_content: memory.data.memory_content,
              memory_index: memory.data.memory_index,
              risk: memory.data.risk || 0
            });
          }
          setActionInfo(null);
          setAgentInfo(null);
          setToolInfo(null);
        } catch (error) {
          console.error('Failed to load memory info:', error);
          setMemoryInfo(null);
        }
      } else if (selectedNode?.type === 'tool_node') {
        try {
          const response = await fetch('/detailed_graph_multi_trace.json');
          const data = await response.json();
          const tool = data.component.nodes.find((t: any) => t.id === selectedNode.id && t.type === 'tool_node');
          if (tool) {
            setToolInfo({
              id: tool.id,
              tool_name: tool.data.tool_name,
              description: tool.data.description,
              risk: tool.data.risk || 0
            });
          }
          setActionInfo(null);
          setAgentInfo(null);
          setMemoryInfo(null);
        } catch (error) {
          console.error('Failed to load tool info:', error);
          setToolInfo(null);
        }
      } else {
        setActionInfo(null);
        setAgentInfo(null);
        setMemoryInfo(null);
        setToolInfo(null);
      }
    };

    loadInfo();
  }, [selectedNode]);

  const onMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    setIsDragging(true);
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isDragging) {
      const newWidth = ((window.innerWidth - e.clientX) / window.innerWidth) * 100;
      setWidth(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, newWidth)));
    }
  }, [isDragging, setWidth]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const formatJsonString = (jsonString: string) => {
    try {
      const parsed = JSON.parse(jsonString);
      return JSON.stringify(parsed, null, 2);
    } catch {
      return jsonString;
    }
  };

  return (
    <div
      className="right-panel"
      ref={panelRef}
      style={{ width: `${width}%` }}
    >
      <div className="right-panel-drag-handle" onMouseDown={onMouseDown} role="presentation" />
      <div className="rp-header">{selectedNode ? selectedNode.data.label : ''}</div>
      
      {actionInfo && (
        <>
          <div className="rp-section">
            <div className="rp-label">Agent Name:</div>
            <div className="rp-value">{actionInfo.agent_name}</div>
            <div className="rp-label">Agent ID:</div>
            <div className="rp-value">{actionInfo.agent_id}</div>
            <div className="rp-label">Jailbreak Success Rate:</div>
            <div className="rp-value">{(actionInfo.average_jailbreak_ASR * 100).toFixed(2)}%</div>
            <div className="rp-label">Blast Radius:</div>
            <div className="rp-value">{actionInfo.blast_radius}</div>
            <div className="rp-label">Weighted Blast Radius:</div>
            <div className="rp-value">{actionInfo.weighted_blast_radius}</div>
            <div className="rp-label">Systemic Risk:</div>
            <div className="rp-value">{(actionInfo.systemic_risk * 100).toFixed(2)}%</div>
            <div className="rp-label">Weighted Systemic Risk:</div>
            <div className="rp-value">{(actionInfo.weighted_systemic_risk * 100).toFixed(2)}%</div>
          </div>

          <div className="rp-section">
            <div className="rp-label">Components Used:</div>
            <div className="rp-box">
              <div className="rp-components-section">
                <div className="rp-components-header">Input Components:</div>
                {actionInfo.input_components.map((componentId: string, index: number) => {
                  const component = componentMap[componentId];
                  return component ? (
                    <div key={index} className="rp-component-item">
                      <span className="rp-component-type">{component.type}:</span>
                      <span className="rp-component-name">{component.name}</span>
                    </div>
                  ) : null;
                })}
              </div>
              <div className="rp-components-section">
                <div className="rp-components-header">Output Components:</div>
                {actionInfo.output_components.map((componentId: string, index: number) => {
                  const component = componentMap[componentId];
                  return component ? (
                    <div key={index} className="rp-component-item">
                      <span className="rp-component-type">{component.type}:</span>
                      <span className="rp-component-name">{component.name}</span>
                    </div>
                  ) : null;
                })}
              </div>
            </div>

            <div className="rp-label" style={{ marginTop: '20px' }}>Input Messages:</div>
            <div className="rp-box" style={{ minHeight: 100 }}>
              {actionInfo.input.map((message: Message, index: number) => (
                <div key={index} className="rp-message-item">
                  <div className="rp-message-type">{message.type}</div>
                  <div className="rp-message-content">
                    {message.content || (message.type === 'ai' && 
                      (message.tool_calls?.length > 0 || message.additional_kwargs?.tool_calls?.length > 0)
                      ? `Calling tool: ${message.tool_calls?.[0]?.name || message.additional_kwargs?.tool_calls?.[0]?.function?.name}`
                      : message.content)
                    }
                  </div>
                  {(message.tool_calls || message.additional_kwargs?.tool_calls) && (
                    <div className="rp-tool-calls">
                      {(message.tool_calls || message.additional_kwargs?.tool_calls)?.map((call, idx) => {
                        // Get tool name and args based on message type
                        const toolName = call?.name || call?.function?.name;
                        const toolArgs = call?.args || call?.function?.arguments;
                        
                        return (
                          <div key={idx} className="rp-tool-call">
                            <span className="rp-tool-name">{toolName}</span>
                            {toolArgs && Object.keys(toolArgs).length > 0 && (
                              <pre className="rp-tool-args">
                                {JSON.stringify(toolArgs, null, 2)}
                              </pre>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="rp-arrow">▼</div>

            <div className="rp-label">Output Message:</div>
            <div className="rp-box" style={{ minHeight: 100 }}>
              {actionInfo.output.generations[0]?.[0]?.message && (
                <div className="rp-message-item">
                  <div className="rp-message-content">
                    {actionInfo.output.generations[0][0].message.content || 
                     (actionInfo.output.generations[0][0].message.additional_kwargs?.tool_calls?.length > 0
                      ? `Calling tool: ${actionInfo.output.generations[0][0].message.additional_kwargs.tool_calls[0]?.function?.name}`
                      : actionInfo.output.generations[0][0].message.content)
                    }
                  </div>
                  {actionInfo.output.generations[0][0].message.additional_kwargs?.tool_calls && (
                    <div className="rp-tool-calls">
                      {actionInfo.output.generations[0][0].message.additional_kwargs?.tool_calls?.map((call: any, idx: number) => {
                        const toolName = call?.function?.name;
                        const toolArgs = call?.function?.arguments;
                        
                        return (
                          <div key={idx} className="rp-tool-call">
                            <span className="rp-tool-name">{toolName}</span>
                            {toolArgs && Object.keys(toolArgs).length > 0 && (
                              <pre className="rp-tool-args">
                                {JSON.stringify(toolArgs, null, 2)}
                              </pre>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {agentInfo && (
        <>
          <div className="rp-section">
            <div className="rp-label">Name:</div>
            <div className="rp-value">{agentInfo.name}</div>
            <div className="rp-label">Model:</div>
            <div className="rp-value">{agentInfo.model}</div>
            <div className="rp-label">Risk Score:</div>
            <div className="rp-value">{(agentInfo.risk * 100).toFixed(2)}%</div>
          </div>

          <div className="rp-section">
            <div className="rp-label">Backstory:</div>
            <div className="rp-box" style={{ minHeight: 100 }}>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {agentInfo.backstory}
              </pre>
            </div>
          </div>

          <div className="rp-section">
            <div className="rp-label">Goal:</div>
            <div className="rp-box" style={{ minHeight: 100 }}>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {agentInfo.goal}
              </pre>
            </div>
          </div>
        </>
      )}

      {memoryInfo && (
        <div className="rp-section">
          <div className="rp-label">Memory Content:</div>
          <div className="rp-box" style={{ minHeight: 100 }}>
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
              {memoryInfo.memory_content}
            </pre>
          </div>
          <div className="rp-label">Memory Index:</div>
          <div className="rp-value">{memoryInfo.memory_index}</div>
          <div className="rp-label">Risk Score:</div>
          <div className="rp-value">{(memoryInfo.risk * 100).toFixed(2)}%</div>
        </div>
      )}

      {toolInfo && (
        <>
          <div className="rp-section">
            <div className="rp-label">Name:</div>
            <div className="rp-value">{toolInfo.tool_name}</div>
            <div className="rp-label">Risk Score:</div>
            <div className="rp-value">{(toolInfo.risk * 100).toFixed(2)}%</div>
          </div>

          <div className="rp-section">
            <div className="rp-label">Description:</div>
            <div className="rp-box" style={{ minHeight: 100 }}>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {toolInfo.description}
              </pre>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default RightPanel; 
import React, { useRef, useEffect, useState, useCallback } from 'react';
import './RightPanel.css';

interface RightPanelProps {
  selectedNode: any;
  width: number;
  setWidth: (w: number) => void;
}

interface ToolCall {
  name?: string;
  args?: Record<string, any>;
  function?: {
    name: string;
    arguments: Record<string, any>;
  };
}

interface Message {
  content: string;
  type: string;
  name?: string;
  tool_calls?: ToolCall[];
  additional_kwargs?: {
    tool_calls?: ToolCall[];
  };
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
  model: string;
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
  system_prompt: string;
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
  const [isLoading, setIsLoading] = useState(false);

  // Helper function to safely render content that might be an object
  const renderContent = (content: any): React.ReactNode => {
    if (typeof content === 'string') {
      return content;
    }
    if (content && typeof content === 'object') {
      // If it's an object, render it as formatted JSON in a pre tag
      return <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{JSON.stringify(content, null, 2)}</pre>;
    }
    return String(content || '');
  };
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadInfo = async () => {
      setError(null);
      setIsLoading(true);
      
      if (selectedNode?.type === 'llm_call_node') {
        try {
          // Get graph structure from reactflow_graph_with_multi_trace.json
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

          // Get action details from detailed_graph_langgraph_multi_trace.json
          const detailsResponse = await fetch('/detailed_graph_langgraph_multi_trace.json');
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
                model: graphAction.data.model || 'Unknown Model',
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
          setError('Failed to load action information. Please try again.');
        }
      } else if (selectedNode?.type === 'agent_node') {
        try {
          const response = await fetch('/detailed_graph_langgraph_multi_trace.json');
          const data = await response.json();
          
          const agent = data?.components?.agents?.find((a: any) => a?.label === selectedNode?.id);
          if (agent) {
            setAgentInfo({
              id: agent.label,
              name: agent.name,
              system_prompt: agent.system_prompt,
              model: selectedNode?.data?.model || 'Unknown Model',
              risk: selectedNode?.data?.risk || 0
            });
          }
          setActionInfo(null);
          setMemoryInfo(null);
          setToolInfo(null);
        } catch (error) {
          console.error('Failed to load agent info:', error);
          setAgentInfo(null);
          setError('Failed to load agent information. Please try again.');
        }
      } else if (selectedNode?.type === 'memory_node') {
        try {
          const response = await fetch('/detailed_graph_langgraph_multi_trace.json');
          const data = await response.json();
          const memory = data?.components?.memories?.find((m: any) => m?.label === selectedNode?.id);
          if (memory) {
            setMemoryInfo({
              id: memory.label,
              memory_content: memory.value,
              memory_index: memory.index || 0,
              risk: memory.risk || 0
            });
          }
          setActionInfo(null);
          setAgentInfo(null);
          setToolInfo(null);
        } catch (error) {
          console.error('Failed to load memory info:', error);
          setMemoryInfo(null);
          setError('Failed to load memory information. Please try again.');
        }
      } else if (selectedNode?.type === 'tool_node') {
        try {
          const response = await fetch('/detailed_graph_langgraph_multi_trace.json');
          const data = await response.json();
          // First try to find the tool in the agent's tools
          let tool = null;
          for (const agent of data?.components?.agents || []) {
            tool = agent.tools?.find((t: any) => t?.tool_name === selectedNode?.id);
            if (tool) break;
          }
          if (tool) {
            setToolInfo({
              id: tool.tool_name,
              tool_name: tool.tool_name,
              description: tool.tool_description,
              risk: tool.risk || 0
            });
          }
          setActionInfo(null);
          setAgentInfo(null);
          setMemoryInfo(null);
        } catch (error) {
          console.error('Failed to load tool info:', error);
          setToolInfo(null);
          setError('Failed to load tool information. Please try again.');
        }
      } else {
        setActionInfo(null);
        setAgentInfo(null);
        setMemoryInfo(null);
        setToolInfo(null);
      }
      
      setIsLoading(false);
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
      
      {isLoading ? (
        <div className="rp-loading">
          <div className="rp-loading-spinner"></div>
          Loading component information...
        </div>
      ) : error ? (
        <div className="rp-error">
          <div className="rp-error-icon">⚠️</div>
          {error}
        </div>
      ) : actionInfo && (
        <>
          <div className="rp-section">
            <div className="rp-header-info">
              <div className="rp-header-main">
                <div className="rp-label">Agent Name:</div>
                <div className="rp-value">{actionInfo.agent_name}</div>
                <div className="rp-label">Agent ID:</div>
                <div className="rp-value">{actionInfo.agent_id}</div>
                <div className="rp-label">Model:</div>
                <div className="rp-value">{actionInfo.model || 'Unknown Model'}</div>
              </div>
            </div>

            <div className="rp-content-box">
              <div className="rp-content-header">Safety Metrics</div>
              <div className="rp-content-body">
                <div className="rp-metrics-grid">
                  <div className="rp-metric-item">
                    <div className="rp-metric-label">Jailbreak Success Rate</div>
                    <div className={`rp-metric-value ${actionInfo.average_jailbreak_ASR > 0.7 ? 'high-risk' : actionInfo.average_jailbreak_ASR > 0.3 ? 'medium-risk' : 'low-risk'}`}>
                      {Number(actionInfo.average_jailbreak_ASR).toFixed(3)}
                    </div>
                  </div>
                  <div className="rp-metric-item">
                    <div className="rp-metric-label">Blast Radius</div>
                    <div className="rp-metric-value">
                      {Number(actionInfo.blast_radius).toFixed(3)}
                    </div>
                  </div>
                  <div className="rp-metric-item">
                    <div className="rp-metric-label">Weighted Blast Radius</div>
                    <div className="rp-metric-value">
                      {Number(actionInfo.weighted_blast_radius).toFixed(3)}
                    </div>
                  </div>
                  <div className="rp-metric-item">
                    <div className="rp-metric-label">Systemic Risk</div>
                    <div className={`rp-metric-value ${actionInfo.systemic_risk > 0.7 ? 'high-risk' : actionInfo.systemic_risk > 0.3 ? 'medium-risk' : 'low-risk'}`}>
                      {Number(actionInfo.systemic_risk).toFixed(3)}
                    </div>
                  </div>
                  <div className="rp-metric-item">
                    <div className="rp-metric-label">Weighted Systemic Risk</div>
                    <div className={`rp-metric-value ${actionInfo.weighted_systemic_risk > 0.7 ? 'high-risk' : actionInfo.weighted_systemic_risk > 0.3 ? 'medium-risk' : 'low-risk'}`}>
                      {Number(actionInfo.weighted_systemic_risk).toFixed(3)}
                    </div>
                  </div>
                </div>
              </div>
            </div>
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
                    {message.content ? renderContent(message.content) : (message.type === 'ai' && 
                      ((message.tool_calls && message.tool_calls.length > 0) || (message.additional_kwargs?.tool_calls && message.additional_kwargs.tool_calls.length > 0))
                      ? `Calling tool: ${message.tool_calls?.[0]?.name || message.additional_kwargs?.tool_calls?.[0]?.function?.name || 'Unknown Tool'}`
                      : renderContent(message.content))
                    }
                  </div>
                  {((message.tool_calls && message.tool_calls.length > 0) || (message.additional_kwargs?.tool_calls && message.additional_kwargs.tool_calls.length > 0)) && (
                    <div className="rp-tool-calls">
                      {(message.tool_calls || message.additional_kwargs?.tool_calls || []).map((call: ToolCall, idx: number) => {
                        // Get tool name and args based on message type
                        const toolName = call?.name || call?.function?.name || 'Unknown Tool';
                        const toolArgs = call?.args || call?.function?.arguments || {};
                        
                        return (
                          <div key={idx} className="rp-tool-call">
                            <span className="rp-tool-name">{toolName}</span>
                            {Object.keys(toolArgs).length > 0 && (
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
              {actionInfo.output.generations?.[0]?.[0]?.message && (
                <div className="rp-message-item">
                  <div className="rp-message-content">
                    {actionInfo.output.generations[0][0].message.content ? 
                      renderContent(actionInfo.output.generations[0][0].message.content) : 
                     (actionInfo.output.generations[0][0].message.additional_kwargs?.tool_calls?.length
                      ? `Calling tool: ${actionInfo.output.generations[0][0].message.additional_kwargs.tool_calls[0]?.function?.name || 'Unknown Tool'}`
                      : renderContent(actionInfo.output.generations[0][0].message.content))
                    }
                  </div>
                  {actionInfo.output.generations[0][0].message.additional_kwargs?.tool_calls?.length && (
                    <div className="rp-tool-calls">
                      {actionInfo.output.generations[0][0].message.additional_kwargs.tool_calls.map((call: ToolCall, idx: number) => {
                        const toolName = call?.function?.name || 'Unknown Tool';
                        const toolArgs = call?.function?.arguments || {};
                        
                        return (
                          <div key={idx} className="rp-tool-call">
                            <span className="rp-tool-name">{toolName}</span>
                            {Object.keys(toolArgs).length > 0 && (
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
            <div className="rp-header-info">
              <div className="rp-header-main">
                <div className="rp-label">Agent Name:</div>
                <div className="rp-value">{agentInfo.name}</div>
              </div>
              <div className="rp-header-stats">
                <div className="rp-stat">
                  <div className="rp-stat-label">Model:</div>
                  <div className="rp-stat-value">{agentInfo.model}</div>
                </div>
                <div className="rp-stat">
                  <div className="rp-stat-label">Risk Score:</div>
                  <div className={`rp-stat-value ${agentInfo.risk > 0.7 ? 'high-risk' : agentInfo.risk > 0.3 ? 'medium-risk' : 'low-risk'}`}>
                    {Number(agentInfo.risk).toFixed(3)}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="rp-section">
            <div className="rp-content-box">
              <div className="rp-content-header">System Prompt</div>
              <div className="rp-content-body">
                <pre>{agentInfo.system_prompt}</pre>
              </div>
            </div>
          </div>
        </>
      )}

      {memoryInfo && (
        <div className="rp-section">
          <div className="rp-header-info">
            <div className="rp-header-main">
              <div className="rp-label">Memory Index:</div>
              <div className="rp-value">{memoryInfo.memory_index}</div>
            </div>
            <div className="rp-header-stats">
              <div className="rp-stat">
                <div className="rp-stat-label">Risk Score:</div>
                <div className={`rp-stat-value ${memoryInfo.risk > 0.7 ? 'high-risk' : memoryInfo.risk > 0.3 ? 'medium-risk' : 'low-risk'}`}>
                  {Number(memoryInfo.risk).toFixed(3)}
                </div>
              </div>
            </div>
          </div>

          <div className="rp-content-box">
            <div className="rp-content-header">Memory Content</div>
            <div className="rp-content-body">
              <pre>{memoryInfo.memory_content}</pre>
            </div>
          </div>
        </div>
      )}

      {toolInfo && (
        <div className="rp-section">
          <div className="rp-header-info">
            <div className="rp-header-main">
              <div className="rp-label">Tool Name:</div>
              <div className="rp-value">{toolInfo.tool_name}</div>
            </div>
            <div className="rp-header-stats">
              <div className="rp-stat">
                <div className="rp-stat-label">Risk Score:</div>
                <div className={`rp-stat-value ${toolInfo.risk > 0.7 ? 'high-risk' : toolInfo.risk > 0.3 ? 'medium-risk' : 'low-risk'}`}>
                  {Number(toolInfo.risk).toFixed(3)}
                </div>
              </div>
            </div>
          </div>

          <div className="rp-content-box">
            <div className="rp-content-header">Description</div>
            <div className="rp-content-body">
              <pre>{toolInfo.description}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RightPanel; 
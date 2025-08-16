'use client';
import React, { useState, useCallback, useEffect } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  useReactFlow,
  Panel,
  Node,
  ReactFlowInstance,
  BackgroundVariant,
  Connection,
  Edge,
  NodeMouseHandler,
  OnNodesChange,
  OnEdgesChange,
  OnInit
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import genericLLMNode from './genericLLMNode';
import AgentNode from './agentNode';
import MemoryNode from './memoryNode';
import ToolNode from './toolNode';
import HumanInputNode from './humanInputNode';
import RightPanel from './RightPanel';

const flowKey = 'example-flow';

let id = 3;
const getId = () => `${id++}`;

interface ActionNodeData extends Record<string, unknown> {
  label: string;
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

function Flow() {
  const [actionNodes, setActionNodes, onActionNodesChange] = useNodesState<Node<Record<string, unknown>, string>>([]);
  const [actionEdges, setActionEdges, onActionEdgesChange] = useEdgesState<Edge<Record<string, unknown>>>([]);
  
  const [componentNodes, setComponentNodes, onComponentNodesChange] = useNodesState<Node<Record<string, unknown>, string>>([]);
  const [componentEdges, setComponentEdges, onComponentEdgesChange] = useEdgesState<Edge<Record<string, unknown>>>([]);
  const [selectedNode, setSelectedNode] = useState<Node<Record<string, unknown>, string> | null>(null);
  const [rightPanelWidth, setRightPanelWidth] = useState(20); // Default to 20% width (minimum)
  const [leftPanelWidth, setLeftPanelWidth] = useState(50); // Default width for the left panel
  const [isDragging, setIsDragging] = useState(false);
  const [highlightedComponents, setHighlightedComponents] = useState<string[]>([]);
  const [showInputComponents, setShowInputComponents] = useState(true); // Toggle between input and output components

  // Function to update highlighted components based on current toggle state and selected node
  const updateHighlightedComponents = useCallback(() => {
    if (selectedNode && selectedNode.type === 'llm_call_node') {
      const nodeData = selectedNode.data as unknown as ActionNodeData;
      if (showInputComponents) {
        const inputComponents = nodeData.input_components || [];
        const agentId = nodeData.agent_id;
        setHighlightedComponents([...inputComponents, agentId]);
      } else {
        const outputComponents = nodeData.output_components || [];
        const agentId = nodeData.agent_id;
        setHighlightedComponents([...outputComponents, agentId]);
      }
    }
  }, [selectedNode, showInputComponents]);

  // Update highlighted components when toggle state changes
  useEffect(() => {
    updateHighlightedComponents();
  }, [updateHighlightedComponents]);

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const response = await fetch('/reactflow_graph_with_multi_trace.json');
        const data = await response.json();
        // Combine nodes and edges from both component and action
        const actionNodes = data.action.nodes.map((node: Node<Record<string, unknown>, string>) => ({
          ...node,
          style: {
            ...node.style,
            opacity: selectedNode != null ? (((node.id) === selectedNode.id) ? 1 : 0.3) : 1,
            transition: 'opacity 0.3s ease',
          },
        }));

        const componentNodes = data.component.nodes.map((node: Node<Record<string, unknown>, string>) => ({
          ...node,
          style: {
            ...node.style,
            opacity: highlightedComponents.length > 0 ? (highlightedComponents.includes(node.id) ? 1 : 0.1) : 1,
            transition: 'opacity 0.3s ease',
          },
        }));
        
        const actionEdges = data.action.edges.map((edge: Edge<Record<string, unknown>>) => ({
          ...edge,
          animated: selectedNode != null ? (((edge.source) === selectedNode.data.label) ? true : false) : false,
          style: {
            ...edge.style,
            stroke: selectedNode != null && edge.source === selectedNode.data.label ? '#0000FF' : '#AFAFAF',
            strokeWidth: selectedNode != null && edge.source === selectedNode.data.label ? 2 : 1,
            opacity: selectedNode ? (edge.source === selectedNode.data.label ? 1 : 0.3) : 1,
            transition: 'stroke 0.3s ease'
          },
        }));

        const componentEdges = data.component.edges.map((edge: Edge<Record<string, unknown>>) => ({
          ...edge,
          style: {
            ...edge.style,
            opacity: highlightedComponents.length > 0 ? (highlightedComponents.includes(edge.source) && highlightedComponents.includes(edge.target)) ? 1 : 0.2 : 1,
            stroke: highlightedComponents.length > 0 ? (highlightedComponents.includes(edge.source) && highlightedComponents.includes(edge.target)) ? '#0000FF' : '#AFAFAF' : '#AFAFAF',
            strokeWidth: highlightedComponents.length > 0 ? (highlightedComponents.includes(edge.source) && highlightedComponents.includes(edge.target)) ? 2 : 1 : 1,
            transition: 'stroke 0.3s ease'
          },
          animated: highlightedComponents.length > 0 ? (highlightedComponents.includes(edge.source) && highlightedComponents.includes(edge.target)) ? true : false : false,
        }));
        
        setActionNodes(actionNodes);
        setActionEdges(actionEdges);
        setComponentNodes(componentNodes);
        setComponentEdges(componentEdges);
      } catch (error) {
        console.error('Failed to load initial flow data:', error);
      }
    };

    loadInitialData();
  }, []);

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        // Get target nodes of edges that have selectedNode as source
        const targetNodeIds = actionEdges
          .filter(edge => edge.source === selectedNode?.data.label)
          .map(edge => edge.target);

        // Create set of active nodes (selected node + target nodes)
        const activeNodeIds = new Set([selectedNode?.id, ...targetNodeIds]);

        // Update action nodes with opacity changes
        setActionNodes(nodes => nodes.map(node => ({
          ...node,
          style: {
            ...node.style,
            opacity: selectedNode ? (activeNodeIds.has(node.id) ? 1 : 0.3) : 1,
            transition: 'opacity 0.3s ease',
          },
        })));

        const actionEdges_ = actionEdges.map(edge => ({
          ...edge,
          animated: selectedNode != null ? (((edge.source) === selectedNode.data.label) ? true : false) : false,
          style: {
            ...edge.style,
            stroke: selectedNode != null && edge.source === selectedNode.data.label ? '#0000FF' : '#AFAFAF',
            strokeWidth: selectedNode != null && edge.source === selectedNode.data.label ? 2 : 1,
            opacity: selectedNode ? (edge.source === selectedNode.data.label ? 1 : 0.3) : 1,
            transition: 'stroke 0.3s ease'
          },
        }));

        setActionEdges(actionEdges_);

      } catch (error) {
        console.error('Failed to load initial flow data:', error);
      }
    };

    loadInitialData();
  }, [selectedNode]);

  useEffect(() => {
    const loadInitialData = async () => {
      try {

        const componentNodes_ = componentNodes.map(node => ({
          ...node,
          style: {
            ...node.style,
            opacity: highlightedComponents.length > 0 ? (highlightedComponents.includes(node.id) ? 1 : 0.1) : 1,
            transition: 'opacity 0.3s ease',
          }
        }));

        const componentEdges_ = componentEdges.map(edge => ({
          ...edge,
          animated: highlightedComponents.length > 0 ? (highlightedComponents.includes(edge.source) && highlightedComponents.includes(edge.target)) ? true : false : false,
          style: {
            ...edge.style,
            opacity: highlightedComponents.length > 0 ? (highlightedComponents.includes(edge.source) && highlightedComponents.includes(edge.target)) ? 1 : 0.2 : 1,
            stroke: highlightedComponents.length > 0 ? (highlightedComponents.includes(edge.source) && highlightedComponents.includes(edge.target)) ? '#0000FF' : '#AFAFAF' : '#AFAFAF',
            strokeWidth: highlightedComponents.length > 0 ? (highlightedComponents.includes(edge.source) && highlightedComponents.includes(edge.target)) ? 2 : 1 : 1,
            transition: 'stroke 0.3s ease',
            animationDirection: showInputComponents ? 'reverse' : 'normal'
          },
          
        }));
        
        setComponentNodes(componentNodes_);
        setComponentEdges(componentEdges_);
      } catch (error) {
        console.error('Failed to load initial flow data:', error);
      }
    };
    loadInitialData();
  }, [highlightedComponents, showInputComponents]);

  const onNodeClick: NodeMouseHandler = useCallback((event, node) => {
    if (node.type === 'llm_call_node') {
      const nodeData = node.data as unknown as ActionNodeData;
      if (showInputComponents) {
        const inputComponents = nodeData.input_components || [];
        const agentId = nodeData.agent_id;
        setHighlightedComponents([...inputComponents, agentId]);
      } else {
        const outputComponents = nodeData.output_components || [];
        const agentId = nodeData.agent_id;
        setHighlightedComponents([...outputComponents, agentId]);
      }
      setSelectedNode(node);
    } else if (node.type === 'agent_node') {
      setHighlightedComponents([]);
      setSelectedNode(node);
    } else if (node.type === 'memory_node') {
      setHighlightedComponents([]);
      setSelectedNode(node);
    } else if (node.type === 'tool_node') {
      setHighlightedComponents([]);
      setSelectedNode(node);
    } else {
      // Clear highlights when clicking any other node type
      setHighlightedComponents([]);
      setSelectedNode(null);
    }
  }, [showInputComponents]);

  const onEdgeClick = useCallback((event: React.MouseEvent) => {
    // Don't dehighlight if clicking on the toggle button
    if ((event.target as Element)?.closest('.component-toggle')) {
      return;
    }
    setHighlightedComponents([]);
    setSelectedNode(null);
  }, []);

  const onPaneClick = useCallback((event: React.MouseEvent) => {
    // Don't dehighlight if clicking on the toggle button
    if ((event.target as Element)?.closest('.component-toggle')) {
      return;
    }
    setHighlightedComponents([]);
    setSelectedNode(null);
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isDragging) {
      const newWidth = (e.clientX / window.innerWidth) * 100;
      setLeftPanelWidth(Math.max(20, Math.min(80, newWidth))); // Limit width between 20% and 80%
    }
  }, [isDragging]);

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

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex' }}>
      <ReactFlowProvider>
        <div style={{ width: `${leftPanelWidth}%`, height: '100%', position: 'relative' }}>
          <ReactFlow
            nodes={[...componentNodes]}
            edges={[...componentEdges]}
            onNodesChange={(changes) => {
              onActionNodesChange(changes);
              onComponentNodesChange(changes);
            }}
            onEdgesChange={(changes) => {
              onActionEdgesChange(changes);
              onComponentEdgesChange(changes);
            }}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            onPaneClick={onPaneClick}
            fitView
            minZoom={0.05}
            nodeTypes={{ 
              llm_call_node: genericLLMNode,
              agent_node: AgentNode,
              memory_node: MemoryNode,
              tool_node: ToolNode,
              human_input_node: HumanInputNode
            }}
            style={{ backgroundColor: '#f9f9f9' }}
          >
            <Controls />
            <MiniMap />
            <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
            <Panel position="top-left" className="component-toggle">
              <button
                onClick={() => setShowInputComponents(!showInputComponents)}
                style={{
                  padding: '8px 16px',
                  backgroundColor: showInputComponents ? '#007bff' : '#ffc107',
                  color: showInputComponents ? 'white' : 'black',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: 'bold',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  transition: 'background-color 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = showInputComponents ? '#0056b3' : '#e6ad06';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = showInputComponents ? '#007bff' : '#ffc107';
                }}
              >
                {showInputComponents ? 'Input Components' : 'Output Components'}
              </button>
            </Panel>
          </ReactFlow>
        </div>
      </ReactFlowProvider>

      <div
        style={{
          width: '4px',
          height: '100%',
          backgroundColor: '#ccc',
          cursor: 'col-resize',
          position: 'relative',
          zIndex: 10,
        }}
        onMouseDown={handleMouseDown}
      />

      <ReactFlowProvider>
        <div style={{ width: `${100 - leftPanelWidth - rightPanelWidth}%`, height: '100%' }}>
          <ReactFlow
            nodes={[...actionNodes]}
            edges={[...actionEdges]}
            onNodesChange={(changes) => {
              onActionNodesChange(changes);
              onComponentNodesChange(changes);
            }}
            onEdgesChange={(changes) => {
              onActionEdgesChange(changes);
              onComponentEdgesChange(changes);
            }}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            onPaneClick={onPaneClick}
            fitView
            minZoom={0.05}
            nodeTypes={{ 
              llm_call_node: genericLLMNode,
              agent_node: AgentNode,
              memory_node: MemoryNode,
              tool_node: ToolNode,
              human_input_node: HumanInputNode
            }}
            style={{ backgroundColor: '#f9f9f9' }}
          >
            <Controls />
            <MiniMap />
            <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
          </ReactFlow>
        </div>
      </ReactFlowProvider>
      <RightPanel selectedNode={selectedNode} width={rightPanelWidth} setWidth={setRightPanelWidth} />
    </div>
  );
}

export default function Page() {
  return (
    <ReactFlowProvider>
      <Flow />
    </ReactFlowProvider>
  );
}

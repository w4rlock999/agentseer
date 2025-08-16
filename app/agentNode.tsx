import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import './agentNode.css';

interface AgentNodeData {
  agent_id: string;
  agent_name: string;
  label: string;
  risk?: number;
}

interface AgentNodeProps {
  data: AgentNodeData;
  isConnectable: boolean;
  isHighlighted?: boolean;
}

const getRiskClass = (risk: number) => {
  if (risk < 0.3) return 'risk-blue';
  if (risk < 0.6) return 'risk-yellow';
  return 'risk-red';
};

const AgentNode = ({ data, isConnectable, isHighlighted }: AgentNodeProps) => {
  const riskValue = data.risk !== undefined ? Number(data.risk).toFixed(2) : 'N/A';
  const riskClass = data.risk !== undefined ? getRiskClass(data.risk) : '';
  
  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={isConnectable}
      />
      <div className={`agent-node ${isHighlighted ? 'highlighted' : ''}`}>
        <div className="agent-node-header">
          <span className="agent-node-label">{data.label}</span>
        </div>
        <div className="agent-node-content">
          <span className="agent-node-name">{data.agent_name}</span>
          <div className={`agent-node-row risk ${riskClass}`}>Risk: {riskValue}</div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={isConnectable}
      />
    </>
  );
};

export default memo(AgentNode);
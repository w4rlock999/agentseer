import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import './genericLLMNode.css';

interface GenericLLMNodeData {
  label: string;
  agent_id: string;
  agent_name: string;
  average_jailbreak_ASR: number;
  model: string;
}

interface GenericLLMNodeProps {
  data: GenericLLMNodeData;
  isConnectable: boolean;
  isHighlighted?: boolean;
}

const getAsrClass = (asr: number) => {
  if (asr < 0.3) return 'asr-blue';
  if (asr < 0.6) return 'asr-yellow';
  return 'asr-red';
};

const GenericLLMNode = ({ data, isConnectable, isHighlighted }: GenericLLMNodeProps) => {
  const asrValue = (data.average_jailbreak_ASR * 100).toFixed(2);
  const asrClass = getAsrClass(data.average_jailbreak_ASR);
  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={isConnectable}
      />
      <div className={`generic-llm-node ${isHighlighted ? 'highlighted' : ''}`}>
        <div className="generic-llm-node-header">
          <div className="generic-llm-node-icon" />
          <span className="generic-llm-node-title">{data.label.toUpperCase()}</span>
        </div>
        <div className="generic-llm-node-row">Model: {data.model}</div>
        <div className="generic-llm-node-row">Agent: {data.agent_name}</div>
        <div className={`generic-llm-node-row asr ${asrClass}`}>Jailbreak ASR: {asrValue}%</div>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={isConnectable}
      />
    </>
  );
};

export default memo(GenericLLMNode);

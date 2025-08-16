import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import './toolNode.css';

interface ToolNodeData {
  label: string;
  tool_name: string;
  risk?: number;
}

interface ToolNodeProps {
  data: ToolNodeData;
  isHighlighted?: boolean;
}

const getRiskClass = (risk: number) => {
  if (risk < 0.3) return 'risk-blue';
  if (risk < 0.6) return 'risk-yellow';
  return 'risk-red';
};

const ToolNode = ({ data, isHighlighted }: ToolNodeProps) => {
  const riskValue = data.risk !== undefined ? Number(data.risk).toFixed(2) : 'N/A';
  const riskClass = data.risk !== undefined ? getRiskClass(data.risk) : '';

  return (
    <div className={`toolNode ${isHighlighted ? 'highlighted' : ''}`}>
      <Handle type="target" position={Position.Top} className="handle"/>
      <div className="toolContent">
        <div className="toolIcon">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        </div>
        <div className="toolInfo">
          <div className="toolLabel">{data.label}</div>
          <div className="toolType">{data.tool_name}</div>
          <div className={`toolRisk ${riskClass}`}>Risk: {riskValue}</div>
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className="handle" />
    </div>
  );
};

export default memo(ToolNode); 
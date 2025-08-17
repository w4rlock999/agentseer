import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import './humanInputNode.css';

interface HumanInputNodeData {
  label: string;
  time: string;
  input: string;
}

function HumanInputNode({ data, isHighlighted }: { data: HumanInputNodeData; isHighlighted?: boolean }) {
  // Format timestamp from "YYYY-MM-DD_HH-MM-SS.SSS" to "MMM DD, YYYY HH:MM:SS"
  const formatTimestamp = (timestamp: string) => {
    const [date, time] = timestamp.split('_');
    const [year, month, day] = date.split('-');
    const [hour, minute, secondWithMs] = time.split('-');
    const second = secondWithMs.split('.')[0];
    
    const dateObj = new Date(Number(year), Number(month) - 1, Number(day), 
                            Number(hour), Number(minute), Number(second));
    
    return dateObj.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  };

  return (
    <div className={`human-input-node ${isHighlighted ? 'highlighted' : ''}`}>
      <div className="human-input-node-header">
        <span>Human Input</span>
        <span className="human-input-node-time">{formatTimestamp(data.time)}</span>
      </div>
      <div className="human-input-node-content">
        {data.input}
      </div>
      {/* Only add a bottom handle since human input nodes only connect downward */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: '#1a73e8', width: '10px', height: '10px' }}
      />
    </div>
  );
}

export default memo(HumanInputNode);
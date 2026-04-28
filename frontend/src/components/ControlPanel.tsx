import React, { useState } from 'react';

interface ControlPanelProps {
  onInject: (text: string) => void;
  onSpeedChange: (multiplier: number) => void;
  onTogglePause: () => void;
  paused: boolean;
  speed: number;
  tier: string;
}

const panelStyle: React.CSSProperties = {
  display: 'flex',
  gap: '12px',
  alignItems: 'flex-start',
  height: '100%',
};

const sectionStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '6px',
};

const labelStyle: React.CSSProperties = {
  fontSize: '10px',
  color: '#666',
  textTransform: 'uppercase',
  letterSpacing: '1px',
};

const inputStyle: React.CSSProperties = {
  background: '#111',
  border: '1px solid #2a2a4e',
  borderRadius: '4px',
  color: '#ccc',
  padding: '6px 8px',
  fontSize: '12px',
  outline: 'none',
  width: '250px',
};

const btnStyle: React.CSSProperties = {
  background: '#2a2a6e',
  color: '#aaa',
  border: '1px solid #3a3a8e',
  borderRadius: '4px',
  padding: '6px 12px',
  fontSize: '11px',
  cursor: 'pointer',
  transition: 'all 0.2s',
};

const disabledBtnStyle: React.CSSProperties = {
  ...btnStyle,
  opacity: 0.4,
  cursor: 'not-allowed',
};

export function ControlPanel({
  onInject, onSpeedChange, onTogglePause, paused, speed, tier,
}: ControlPanelProps) {
  const [eventText, setEventText] = useState('');
  const canInject = tier !== 'free';
  const canSpeed = tier !== 'free';

  const handleInject = () => {
    if (eventText.trim() && canInject) {
      onInject(eventText.trim());
      setEventText('');
    }
  };

  return (
    <div style={panelStyle}>
      <div style={sectionStyle}>
        <div style={labelStyle}>INJECT EVENT</div>
        <div style={{ display: 'flex', gap: '4px' }}>
          <input
            type="text"
            placeholder={canInject ? 'Ask the agent something...' : 'Upgrade to inject events'}
            value={eventText}
            onChange={(e) => setEventText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleInject()}
            style={inputStyle}
            disabled={!canInject}
          />
          <button
            onClick={handleInject}
            style={canInject ? btnStyle : disabledBtnStyle}
            disabled={!canInject}
          >
            Inject
          </button>
        </div>
        {!canInject && (
          <div style={{ fontSize: '9px', color: '#f44336' }}>
            Event injection requires Pro tier or higher
          </div>
        )}
      </div>

      <div style={sectionStyle}>
        <div style={labelStyle}>SIMULATION</div>
        <div style={{ display: 'flex', gap: '4px' }}>
          <button onClick={onTogglePause} style={btnStyle}>
            {paused ? '▶ Resume' : '⏸ Pause'}
          </button>
        </div>
      </div>

      <div style={sectionStyle}>
        <div style={labelStyle}>SPEED: {speed.toFixed(1)}×</div>
        <div style={{ display: 'flex', gap: '4px' }}>
          {[0.5, 1, 2, 5, 10].map((s) => (
            <button
              key={s}
              onClick={() => canSpeed && onSpeedChange(s)}
              style={{
                ...(canSpeed ? btnStyle : disabledBtnStyle),
                background: Math.abs(speed - s) < 0.1 ? '#4040a0' : '#2a2a6e',
                padding: '4px 8px',
              }}
              disabled={!canSpeed}
            >
              {s}×
            </button>
          ))}
        </div>
        {!canSpeed && (
          <div style={{ fontSize: '9px', color: '#f44336' }}>
            Speed control requires Pro tier
          </div>
        )}
      </div>
    </div>
  );
}

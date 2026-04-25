import React from 'react';

interface ActionInfo {
  action?: number[];
  confidence?: number;
  entropy?: number;
  action_mean?: number[];
  action_std?: number[];
  discrete_action?: string;
  discrete_probs?: number[];
}

interface TrainingInfo {
  total_steps?: number;
  episode?: number;
  stats?: {
    policy_loss?: number;
    value_loss?: number;
    entropy?: number;
  };
}

const panelStyle: React.CSSProperties = {
  background: '#1a1a2e',
  borderRadius: '6px',
  padding: '8px',
  marginBottom: '4px',
};

const labelStyle: React.CSSProperties = {
  fontSize: '10px',
  color: '#666',
  textTransform: 'uppercase',
  letterSpacing: '1px',
  marginBottom: '6px',
};

export function DecisionPanel({ action, training }: {
  action?: ActionInfo | null;
  training?: TrainingInfo | null;
}) {
  return (
    <div style={panelStyle}>
      <div style={labelStyle}>DECISION</div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px', marginBottom: '8px' }}>
        {['Forward', 'Strafe', 'Rotation'].map((name, i) => (
          <ActionBar
            key={name}
            label={name}
            value={action?.action?.[i] ?? 0}
          />
        ))}
      </div>

      {action?.discrete_action && (
        <div style={{ marginBottom: '6px' }}>
          <div style={{ fontSize: '9px', color: '#555', marginBottom: '2px' }}>DISCRETE ACTION</div>
          <div style={{ display: 'flex', gap: '3px' }}>
            {['none', 'interact', 'pick', 'drop'].map((a, i) => (
              <span key={a} style={{
                flex: 1, textAlign: 'center', fontSize: '9px', padding: '2px',
                borderRadius: '3px',
                background: action.discrete_action === a ? '#2a4a2a' : '#111',
                color: action.discrete_action === a ? '#4caf50' : '#555',
                border: action.discrete_action === a ? '1px solid #4caf50' : '1px solid #222',
              }}>
                {a.toUpperCase()}
                {action.discrete_probs?.[i] !== undefined && (
                  <div style={{ fontSize: '8px', color: '#444' }}>
                    {(action.discrete_probs[i] * 100).toFixed(0)}%
                  </div>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
        <StatBox label="Confidence" value={`${((action?.confidence ?? 0) * 100).toFixed(0)}%`}
          color={getConfidenceColor(action?.confidence ?? 0)} />
        <StatBox label="Entropy" value={(action?.entropy ?? 0).toFixed(2)} color="#ff9800" />
      </div>

      {training && (
        <>
          <div style={{ ...labelStyle, marginTop: '4px' }}>TRAINING</div>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            <StatBox label="Steps" value={String(training.total_steps ?? 0)} color="#7b7bff" />
            <StatBox label="Episode" value={String(training.episode ?? 0)} color="#7b7bff" />
            <StatBox label="P.Loss"
              value={(training.stats?.policy_loss ?? 0).toFixed(4)} color="#f44336" />
            <StatBox label="V.Loss"
              value={(training.stats?.value_loss ?? 0).toFixed(4)} color="#ff9800" />
          </div>
        </>
      )}
    </div>
  );
}

function ActionBar({ label, value }: { label: string; value: number }) {
  const pct = ((value + 1) / 2) * 100;
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: '9px', color: '#555' }}>{label}</div>
      <div style={{
        background: '#111', borderRadius: '3px', height: '20px',
        position: 'relative', overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute', left: '50%', top: 0, bottom: 0,
          width: '1px', background: '#333',
        }} />
        <div style={{
          position: 'absolute',
          left: value >= 0 ? '50%' : `${pct}%`,
          width: `${Math.abs(value) * 50}%`,
          top: 0, bottom: 0,
          background: value >= 0 ? '#4caf50' : '#f44336',
          borderRadius: '2px',
          transition: 'all 0.1s',
        }} />
      </div>
      <div style={{ fontSize: '10px', color: '#888', fontFamily: 'monospace' }}>
        {value.toFixed(2)}
      </div>
    </div>
  );
}

function StatBox({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      flex: 1, background: '#111', borderRadius: '4px', padding: '4px 6px',
      borderLeft: `2px solid ${color}`,
    }}>
      <div style={{ fontSize: '9px', color: '#555' }}>{label}</div>
      <div style={{ fontSize: '13px', color, fontFamily: 'monospace', fontWeight: 600 }}>
        {value}
      </div>
    </div>
  );
}

function getConfidenceColor(c: number): string {
  if (c >= 0.7) return '#4caf50';
  if (c >= 0.3) return '#ff9800';
  return '#f44336';
}

import React from 'react';

interface MemoryInfo {
  episodic_size?: number;
  replay_size?: number;
}

interface EventsInfo {
  active?: number;
  resolved?: number;
}

const panelStyle: React.CSSProperties = {
  background: '#1a1a2e',
  borderRadius: '6px',
  padding: '8px',
  marginBottom: '4px',
  flex: 1,
  overflow: 'auto',
};

const labelStyle: React.CSSProperties = {
  fontSize: '10px',
  color: '#666',
  textTransform: 'uppercase',
  letterSpacing: '1px',
  marginBottom: '6px',
};

export function MemoryTimeline({ memory, events }: {
  memory?: MemoryInfo | null;
  events?: EventsInfo | null;
}) {
  return (
    <div style={panelStyle}>
      <div style={labelStyle}>MEMORY / TIMELINE</div>

      <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
        <StatBox label="Episodic" value={String(memory?.episodic_size ?? 0)} color="#7b7bff" />
        <StatBox label="Replay" value={String(memory?.replay_size ?? 0)} color="#4caf50" />
        <StatBox label="Active Evts" value={String(events?.active ?? 0)} color="#ff9800" />
        <StatBox label="Resolved" value={String(events?.resolved ?? 0)} color="#888" />
      </div>

      <div style={{ fontSize: '10px', color: '#555', textAlign: 'center', padding: '12px 0' }}>
        Memory timeline visualization will appear here as memories accumulate
      </div>
    </div>
  );
}

function StatBox({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      flex: 1, background: '#111', borderRadius: '4px', padding: '4px 6px',
      borderLeft: `2px solid ${color}`, minWidth: '60px',
    }}>
      <div style={{ fontSize: '9px', color: '#555' }}>{label}</div>
      <div style={{ fontSize: '13px', color, fontFamily: 'monospace', fontWeight: 600 }}>
        {value}
      </div>
    </div>
  );
}

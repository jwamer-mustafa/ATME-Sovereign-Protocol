import React from 'react';

interface PerceptionData {
  frame_raw?: string;
  attention_map?: string;
  confidence?: number;
  entropy?: number;
  action_logits?: number[];
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
  marginBottom: '4px',
};

const imgStyle: React.CSSProperties = {
  width: '100%',
  imageRendering: 'pixelated',
  borderRadius: '4px',
  border: '1px solid #2a2a4e',
};

export function PerceptionPanel({ perception }: { perception: PerceptionData | null }) {
  return (
    <div style={panelStyle}>
      <div style={labelStyle}>PERCEPTION</div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', marginBottom: '8px' }}>
        <div>
          <div style={{ ...labelStyle, fontSize: '9px' }}>CAMERA VIEW</div>
          {perception?.frame_raw ? (
            <img
              src={`data:image/jpeg;base64,${perception.frame_raw}`}
              alt="Agent Camera"
              style={imgStyle}
            />
          ) : (
            <div style={{ ...imgStyle, height: '120px', background: '#111', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ color: '#444', fontSize: '10px' }}>No Frame</span>
            </div>
          )}
        </div>

        <div>
          <div style={{ ...labelStyle, fontSize: '9px' }}>ATTENTION MAP</div>
          {perception?.attention_map ? (
            <img
              src={`data:image/jpeg;base64,${perception.attention_map}`}
              alt="Attention Heatmap"
              style={{ ...imgStyle, filter: 'hue-rotate(200deg) saturate(2)' }}
            />
          ) : (
            <div style={{ ...imgStyle, height: '120px', background: '#111', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ color: '#444', fontSize: '10px' }}>No Attention</span>
            </div>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', gap: '8px' }}>
        <Metric label="Confidence" value={perception?.confidence ?? 0} color="#4caf50" />
        <Metric label="Entropy" value={perception?.entropy ?? 0} color="#ff9800" max={5} />
      </div>

      {perception?.action_logits && (
        <div style={{ marginTop: '4px' }}>
          <div style={{ ...labelStyle, fontSize: '9px' }}>ACTION LOGITS</div>
          <div style={{ display: 'flex', gap: '4px' }}>
            {['Fwd/Bwd', 'L/R', 'Rot'].map((name, i) => (
              <div key={name} style={{ flex: 1, textAlign: 'center' }}>
                <div style={{ fontSize: '9px', color: '#555' }}>{name}</div>
                <div style={{ fontSize: '12px', color: '#aaa', fontFamily: 'monospace' }}>
                  {(perception.action_logits![i] ?? 0).toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, color, max = 1 }: {
  label: string; value: number; color: string; max?: number;
}) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ flex: 1 }}>
      <div style={{ fontSize: '9px', color: '#555', marginBottom: '2px' }}>{label}</div>
      <div style={{ background: '#111', borderRadius: '3px', height: '14px', overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: color,
          borderRadius: '3px', transition: 'width 0.3s',
        }} />
      </div>
      <div style={{ fontSize: '10px', color: '#888', textAlign: 'right', fontFamily: 'monospace' }}>
        {value.toFixed(3)}
      </div>
    </div>
  );
}

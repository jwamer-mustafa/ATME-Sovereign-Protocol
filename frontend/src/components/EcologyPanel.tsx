import React from 'react';

interface EcologyState {
  day_night?: {
    time?: number;
    time_of_day?: string;
    sun_intensity?: number;
    sky_color?: number[];
  };
  weather?: {
    type?: string;
    intensity?: number;
  };
  difficulty?: number;
  resources?: Array<{
    id: string;
    type: string;
    collected: boolean;
    value: number;
  }>;
  npcs?: Array<{
    id: string;
    behavior: string;
  }>;
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

export function EcologyPanel({ ecology }: { ecology?: EcologyState | null }) {
  if (!ecology) return null;

  const dn = ecology.day_night;
  const weather = ecology.weather;
  const skyColor = dn?.sky_color
    ? `rgb(${Math.round(dn.sky_color[0] * 255)}, ${Math.round(dn.sky_color[1] * 255)}, ${Math.round(dn.sky_color[2] * 255)})`
    : '#333';

  const activeResources = ecology.resources?.filter(r => !r.collected).length ?? 0;
  const totalResources = ecology.resources?.length ?? 0;

  return (
    <div style={panelStyle}>
      <div style={labelStyle}>ECOLOGY</div>

      <div style={{ display: 'flex', gap: '6px', marginBottom: '6px', flexWrap: 'wrap' }}>
        <InfoBox
          label="Time"
          value={dn?.time_of_day?.toUpperCase() ?? 'N/A'}
          color={getTimeColor(dn?.time_of_day)}
          icon={getTimeIcon(dn?.time_of_day)}
        />
        <InfoBox
          label="Weather"
          value={weather?.type?.toUpperCase() ?? 'CLEAR'}
          color={getWeatherColor(weather?.type)}
          icon={getWeatherIcon(weather?.type)}
        />
        <InfoBox
          label="Difficulty"
          value={`${((ecology.difficulty ?? 0) * 100).toFixed(0)}%`}
          color="#ff9800"
        />
      </div>

      <div style={{ display: 'flex', gap: '6px', marginBottom: '4px' }}>
        <InfoBox
          label="Resources"
          value={`${activeResources}/${totalResources}`}
          color="#4caf50"
        />
        <InfoBox
          label="NPCs"
          value={String(ecology.npcs?.length ?? 0)}
          color="#9c27b0"
        />
        <div style={{
          flex: 1, background: '#111', borderRadius: '4px', padding: '4px 6px',
          display: 'flex', alignItems: 'center', gap: '4px',
        }}>
          <div style={{ fontSize: '9px', color: '#555' }}>Sky</div>
          <div style={{
            width: '20px', height: '14px', borderRadius: '2px',
            background: skyColor, border: '1px solid #333',
          }} />
        </div>
      </div>

      {dn && (
        <div style={{
          background: '#111', borderRadius: '3px', height: '6px',
          overflow: 'hidden', marginTop: '4px',
        }}>
          <div style={{
            width: `${(dn.time ?? 0) * 100}%`,
            height: '100%',
            background: `linear-gradient(90deg, #1a237e, #ff9800, #ffeb3b, #ff9800, #1a237e)`,
            transition: 'width 0.5s',
          }} />
        </div>
      )}
    </div>
  );
}

function InfoBox({ label, value, color, icon }: {
  label: string; value: string; color: string; icon?: string;
}) {
  return (
    <div style={{
      flex: 1, background: '#111', borderRadius: '4px', padding: '4px 6px',
      borderLeft: `2px solid ${color}`,
    }}>
      <div style={{ fontSize: '9px', color: '#555' }}>{label}</div>
      <div style={{ fontSize: '12px', color, fontFamily: 'monospace', fontWeight: 600 }}>
        {icon && <span style={{ marginRight: '2px' }}>{icon}</span>}
        {value}
      </div>
    </div>
  );
}

function getTimeColor(tod?: string): string {
  switch (tod) {
    case 'dawn': return '#ff9800';
    case 'day': return '#ffeb3b';
    case 'dusk': return '#e65100';
    case 'night': return '#1a237e';
    default: return '#666';
  }
}

function getTimeIcon(tod?: string): string {
  switch (tod) {
    case 'dawn': case 'dusk': return '\u263C';
    case 'day': return '\u2600';
    case 'night': return '\u263E';
    default: return '';
  }
}

function getWeatherColor(w?: string): string {
  switch (w) {
    case 'rain': return '#42a5f5';
    case 'storm': return '#ef5350';
    case 'fog': return '#78909c';
    default: return '#4caf50';
  }
}

function getWeatherIcon(w?: string): string {
  switch (w) {
    case 'rain': return '\u2602';
    case 'storm': return '\u26A1';
    case 'fog': return '\u2601';
    default: return '\u2600';
  }
}

import React, { useState } from 'react';

interface Badge {
  id: string;
  name: string;
  description: string;
  skill_level: number;
  category: string;
}

interface AgentProfile {
  id: string;
  name: string;
  style: string;
  description: string;
  traits: Record<string, number>;
}

interface RetentionState {
  user_id?: string;
  interaction_count?: number;
  badges?: Badge[];
  selected_agent?: string;
  available_agents?: AgentProfile[];
  evolution?: {
    has_history?: boolean;
    current?: { reward?: number; episodes?: number; badges?: number };
    daily_change?: { reward_delta?: number; new_badges?: number };
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

export function RetentionPanel({ retention, onSelectAgent }: {
  retention?: RetentionState | null;
  onSelectAgent?: (agentId: string) => void;
}) {
  const [showAgents, setShowAgents] = useState(false);

  if (!retention) return null;

  return (
    <div style={panelStyle}>
      <div style={labelStyle}>RETENTION & AGENTS</div>

      {/* Agent selector */}
      <div style={{ marginBottom: '6px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '11px', color: '#888' }}>
            Agent: {getAgentName(retention)}
          </span>
          <button
            onClick={() => setShowAgents(!showAgents)}
            style={smallBtnStyle}
          >
            {showAgents ? 'Hide' : 'Change'}
          </button>
        </div>

        {showAgents && retention.available_agents && (
          <div style={{ marginTop: '4px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
            {retention.available_agents.map(agent => (
              <div
                key={agent.id}
                onClick={() => onSelectAgent?.(agent.id)}
                style={{
                  background: agent.id === retention.selected_agent ? '#2a2a5e' : '#111',
                  border: agent.id === retention.selected_agent ? '1px solid #7b7bff' : '1px solid #222',
                  borderRadius: '4px',
                  padding: '4px 6px',
                  cursor: 'pointer',
                  fontSize: '11px',
                }}
              >
                <div style={{ color: '#ccc', fontWeight: 600 }}>{agent.name}</div>
                <div style={{ color: '#666', fontSize: '9px' }}>{agent.description}</div>
                <div style={{ display: 'flex', gap: '4px', marginTop: '2px' }}>
                  {Object.entries(agent.traits).map(([trait, val]) => (
                    <TraitBar key={trait} name={trait} value={val} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Badges */}
      {retention.badges && retention.badges.length > 0 && (
        <div style={{ marginBottom: '6px' }}>
          <div style={{ fontSize: '9px', color: '#555', marginBottom: '2px' }}>BADGES</div>
          <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
            {retention.badges.map(badge => (
              <span key={badge.id} title={badge.description} style={{
                background: '#2a2a4e',
                color: '#aad',
                fontSize: '9px',
                padding: '2px 5px',
                borderRadius: '3px',
                border: '1px solid #3a3a6e',
              }}>
                {badge.name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Evolution */}
      {retention.evolution?.has_history && (
        <div>
          <div style={{ fontSize: '9px', color: '#555', marginBottom: '2px' }}>EVOLUTION</div>
          <div style={{ display: 'flex', gap: '6px' }}>
            <StatMini
              label="Reward"
              value={retention.evolution.current?.reward?.toFixed(1) ?? '0'}
              delta={retention.evolution.daily_change?.reward_delta}
            />
            <StatMini
              label="Interactions"
              value={String(retention.interaction_count ?? 0)}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function TraitBar({ name, value }: { name: string; value: number }) {
  return (
    <div style={{ flex: 1, minWidth: '40px' }}>
      <div style={{ fontSize: '8px', color: '#444', textTransform: 'capitalize' }}>{name}</div>
      <div style={{ background: '#111', height: '4px', borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{
          width: `${value * 100}%`,
          height: '100%',
          background: '#7b7bff',
          borderRadius: '2px',
        }} />
      </div>
    </div>
  );
}

function StatMini({ label, value, delta }: {
  label: string; value: string; delta?: number;
}) {
  return (
    <div style={{ flex: 1, background: '#111', borderRadius: '3px', padding: '3px 5px' }}>
      <div style={{ fontSize: '8px', color: '#555' }}>{label}</div>
      <div style={{ fontSize: '11px', color: '#aaa', fontFamily: 'monospace' }}>
        {value}
        {delta !== undefined && (
          <span style={{
            fontSize: '9px', marginLeft: '3px',
            color: delta >= 0 ? '#4caf50' : '#f44336',
          }}>
            {delta >= 0 ? '+' : ''}{delta.toFixed(1)}
          </span>
        )}
      </div>
    </div>
  );
}

function getAgentName(retention: RetentionState): string {
  const selected = retention.available_agents?.find(a => a.id === retention.selected_agent);
  return selected?.name ?? 'Explorer';
}

const smallBtnStyle: React.CSSProperties = {
  background: '#2a2a4e',
  color: '#aaa',
  border: '1px solid #3a3a5e',
  borderRadius: '3px',
  padding: '1px 6px',
  fontSize: '9px',
  cursor: 'pointer',
};

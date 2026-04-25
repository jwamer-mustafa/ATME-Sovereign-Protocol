import React, { useState, useCallback } from 'react';
import { Scene3D } from './components/Scene3D';
import { PerceptionPanel } from './components/PerceptionPanel';
import { DecisionPanel } from './components/DecisionPanel';
import { ControlPanel } from './components/ControlPanel';
import { MemoryTimeline } from './components/MemoryTimeline';
import { AuthPanel } from './components/AuthPanel';
import { BillingPanel } from './components/BillingPanel';
import { EcologyPanel } from './components/EcologyPanel';
import { RetentionPanel } from './components/RetentionPanel';
import { useWebSocket } from './hooks/useWebSocket';
import { useAuth } from './hooks/useAuth';

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'grid',
    gridTemplateColumns: '1fr 320px',
    gridTemplateRows: '1fr 200px',
    height: '100vh',
    width: '100vw',
    gap: '2px',
    background: '#111118',
  },
  mainView: {
    gridRow: '1',
    gridColumn: '1',
    position: 'relative',
    overflow: 'hidden',
  },
  sidePanel: {
    gridRow: '1 / 3',
    gridColumn: '2',
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    overflow: 'auto',
    background: '#14141e',
    padding: '8px',
  },
  bottomPanel: {
    gridRow: '2',
    gridColumn: '1',
    display: 'flex',
    gap: '2px',
    overflow: 'hidden',
  },
  panelSection: {
    flex: 1,
    background: '#1a1a2e',
    borderRadius: '4px',
    padding: '8px',
    overflow: 'auto',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '4px 8px',
    background: '#0d0d15',
    borderBottom: '1px solid #2a2a3e',
  },
  title: {
    fontSize: '14px',
    fontWeight: 700,
    color: '#7b7bff',
    letterSpacing: '1px',
  },
};

export default function App() {
  const { user, token, login, register, logout } = useAuth();
  const { state, perception, connected, sendMessage } = useWebSocket(token);
  const [showBilling, setShowBilling] = useState(false);
  const [showAuth, setShowAuth] = useState(false);

  const handleInjectEvent = useCallback((text: string) => {
    sendMessage({ type: 'inject_event', text });
  }, [sendMessage]);

  const handleSpeedChange = useCallback((multiplier: number) => {
    sendMessage({ type: 'set_speed', multiplier });
  }, [sendMessage]);

  const handleTogglePause = useCallback(() => {
    sendMessage({ type: 'toggle_pause' });
  }, [sendMessage]);

  const handleSelectAgent = useCallback(async (agentId: string) => {
    if (!token) return;
    try {
      await fetch('/api/retention/select-agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ agent_id: agentId }),
      });
    } catch (e) {
      console.error('Agent selection failed', e);
    }
  }, [token]);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>EMBODIED AI PLATFORM</span>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ fontSize: '11px', color: connected ? '#4caf50' : '#f44336' }}>
            {connected ? '● CONNECTED' : '○ DISCONNECTED'}
          </span>
          {user ? (
            <>
              <span style={{ fontSize: '11px', color: '#888' }}>
                {user.email} [{user.tier.toUpperCase()}]
              </span>
              <button onClick={() => setShowBilling(!showBilling)}
                style={btnStyle}>Billing</button>
              <button onClick={logout} style={btnStyle}>Logout</button>
            </>
          ) : (
            <button onClick={() => setShowAuth(!showAuth)} style={btnStyle}>
              Login / Register
            </button>
          )}
        </div>
      </div>

      {showAuth && !user && (
        <AuthPanel onLogin={login} onRegister={register}
          onClose={() => setShowAuth(false)} />
      )}
      {showBilling && user && (
        <BillingPanel tier={user.tier} token={token}
          onClose={() => setShowBilling(false)} />
      )}

      <div style={styles.mainView}>
        <Scene3D state={state} />
      </div>

      <div style={styles.sidePanel}>
        <PerceptionPanel perception={perception} />
        <DecisionPanel action={state?.action} training={state?.training} />
        <EcologyPanel ecology={state?.ecology} />
        <RetentionPanel retention={state?.retention} onSelectAgent={handleSelectAgent} />
        <MemoryTimeline memory={state?.memory} events={state?.events} />
      </div>

      <div style={styles.bottomPanel}>
        <div style={styles.panelSection}>
          <ControlPanel
            onInject={handleInjectEvent}
            onSpeedChange={handleSpeedChange}
            onTogglePause={handleTogglePause}
            paused={state?.paused ?? false}
            speed={state?.speed ?? 1}
            tier={user?.tier ?? 'free'}
          />
        </div>
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  background: '#2a2a4e',
  color: '#aaa',
  border: '1px solid #3a3a5e',
  borderRadius: '3px',
  padding: '2px 8px',
  fontSize: '11px',
  cursor: 'pointer',
};

import React, { useEffect, useState } from 'react';

interface Tier {
  id: string;
  name: string;
  price_monthly: number;
  events_per_day: number;
  live_stream: boolean;
  env_control: boolean;
  speed_control: boolean;
}

interface BillingPanelProps {
  tier: string;
  token: string | null;
  onClose: () => void;
}

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0, left: 0, right: 0, bottom: 0,
  background: 'rgba(0,0,0,0.7)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 1000,
};

const panelStyle: React.CSSProperties = {
  background: '#1a1a2e',
  borderRadius: '8px',
  padding: '24px',
  width: '600px',
  border: '1px solid #2a2a4e',
};

export function BillingPanel({ tier, token, onClose }: BillingPanelProps) {
  const [tiers, setTiers] = useState<Tier[]>([]);

  useEffect(() => {
    fetch('/api/billing/tiers')
      .then((r) => r.json())
      .then((data) => setTiers(data.tiers))
      .catch(() => {});
  }, []);

  const handleUpgrade = async (tierId: string) => {
    if (!token) return;
    try {
      const res = await fetch(`/api/billing/upgrade?tier=${tierId}`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (data.access_token) {
        localStorage.setItem('eai_token', data.access_token);
      }
      window.location.reload();
    } catch (e) {
      console.error('Upgrade failed', e);
    }
  };

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={panelStyle} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ color: '#7b7bff', marginBottom: '16px' }}>Subscription Plans</h3>
        <div style={{ display: 'flex', gap: '12px' }}>
          {tiers.map((t) => (
            <div
              key={t.id}
              style={{
                flex: 1,
                background: t.id === tier ? '#2a2a5e' : '#111',
                borderRadius: '8px',
                padding: '16px',
                border: t.id === tier ? '2px solid #7b7bff' : '1px solid #2a2a4e',
              }}
            >
              <div style={{ fontSize: '16px', fontWeight: 700, color: '#ddd', marginBottom: '4px' }}>
                {t.name}
              </div>
              <div style={{ fontSize: '24px', color: '#7b7bff', marginBottom: '12px' }}>
                ${t.price_monthly}<span style={{ fontSize: '12px', color: '#666' }}>/mo</span>
              </div>
              <ul style={{ listStyle: 'none', padding: 0, fontSize: '11px', color: '#888' }}>
                <li>{t.live_stream ? '● Live stream' : '● Delayed stream (5s)'}</li>
                <li>● {t.events_per_day > 100 ? 'Unlimited' : t.events_per_day} events/day</li>
                <li>{t.env_control ? '● Environment control' : '○ No env control'}</li>
                <li>{t.speed_control ? '● Speed control' : '○ No speed control'}</li>
              </ul>
              {t.id !== tier && t.id !== 'free' && (
                <button
                  onClick={() => handleUpgrade(t.id)}
                  style={{
                    width: '100%',
                    marginTop: '8px',
                    background: '#3a3a8e',
                    color: '#ddd',
                    border: 'none',
                    borderRadius: '4px',
                    padding: '6px',
                    cursor: 'pointer',
                    fontSize: '12px',
                  }}
                >
                  Upgrade
                </button>
              )}
              {t.id === tier && (
                <div style={{
                  textAlign: 'center', marginTop: '8px', fontSize: '11px',
                  color: '#4caf50',
                }}>
                  Current Plan
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

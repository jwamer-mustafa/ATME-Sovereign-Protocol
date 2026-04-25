import React, { useState } from 'react';

interface AuthPanelProps {
  onLogin: (email: string, password: string) => Promise<void>;
  onRegister: (email: string, password: string) => Promise<void>;
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

const formStyle: React.CSSProperties = {
  background: '#1a1a2e',
  borderRadius: '8px',
  padding: '24px',
  width: '320px',
  border: '1px solid #2a2a4e',
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  background: '#111',
  border: '1px solid #2a2a4e',
  borderRadius: '4px',
  color: '#ccc',
  padding: '8px 12px',
  fontSize: '13px',
  marginBottom: '8px',
  outline: 'none',
};

const btnStyle: React.CSSProperties = {
  width: '100%',
  background: '#3a3a8e',
  color: '#ddd',
  border: 'none',
  borderRadius: '4px',
  padding: '8px',
  fontSize: '13px',
  cursor: 'pointer',
  marginBottom: '6px',
};

export function AuthPanel({ onLogin, onRegister, onClose }: AuthPanelProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isRegister, setIsRegister] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    try {
      setError('');
      if (isRegister) {
        await onRegister(email, password);
      } else {
        await onLogin(email, password);
      }
      onClose();
    } catch (e: any) {
      setError(e.message || 'Authentication failed');
    }
  };

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={formStyle} onClick={(e) => e.stopPropagation()}>
        <h3 style={{ color: '#7b7bff', marginBottom: '16px', fontSize: '16px' }}>
          {isRegister ? 'Register' : 'Login'}
        </h3>
        {error && (
          <div style={{ color: '#f44336', fontSize: '12px', marginBottom: '8px' }}>{error}</div>
        )}
        <input
          type="email" placeholder="Email" value={email}
          onChange={(e) => setEmail(e.target.value)} style={inputStyle}
        />
        <input
          type="password" placeholder="Password" value={password}
          onChange={(e) => setPassword(e.target.value)} style={inputStyle}
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
        />
        <button onClick={handleSubmit} style={btnStyle}>
          {isRegister ? 'Register' : 'Login'}
        </button>
        <button
          onClick={() => setIsRegister(!isRegister)}
          style={{ ...btnStyle, background: 'transparent', color: '#888' }}
        >
          {isRegister ? 'Already have an account? Login' : 'Need an account? Register'}
        </button>
      </div>
    </div>
  );
}

import { useState, useCallback, useEffect } from 'react';

interface User {
  user_id: string;
  email: string;
  tier: string;
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(
    localStorage.getItem('eai_token')
  );

  useEffect(() => {
    if (token) {
      fetch('/api/auth/me', {
        headers: { Authorization: `Bearer ${token}` },
      })
        .then((r) => {
          if (!r.ok) throw new Error('Invalid token');
          return r.json();
        })
        .then(setUser)
        .catch(() => {
          setToken(null);
          setUser(null);
          localStorage.removeItem('eai_token');
        });
    }
  }, [token]);

  const login = useCallback(async (email: string, password: string) => {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || 'Login failed');
    }
    const data = await res.json();
    setToken(data.access_token);
    localStorage.setItem('eai_token', data.access_token);
  }, []);

  const register = useCallback(async (email: string, password: string) => {
    const res = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.detail || 'Registration failed');
    }
    const data = await res.json();
    setToken(data.access_token);
    localStorage.setItem('eai_token', data.access_token);
  }, []);

  const logout = useCallback(() => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('eai_token');
  }, []);

  return { user, token, login, register, logout };
}

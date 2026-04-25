import { useCallback, useEffect, useRef, useState } from 'react';

interface SimState {
  environment?: any;
  training?: any;
  memory?: any;
  events?: any;
  action?: any;
  speed?: number;
  paused?: boolean;
}

interface PerceptionData {
  frame_raw?: string;
  attention_map?: string;
  confidence?: number;
  entropy?: number;
  action_logits?: number[];
}

export function useWebSocket(token: string | null) {
  const [state, setState] = useState<SimState | null>(null);
  const [perception, setPerception] = useState<PerceptionData | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number | null>(null);

  const connect = useCallback(() => {
    const clientId = `client_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    let url = `${protocol}//${host}/ws/${clientId}`;
    if (token) {
      url += `?token=${token}`;
    }

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      // Start ping interval
      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
        }
      }, 10000);
      ws.addEventListener('close', () => clearInterval(pingInterval));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'state') {
          setState(msg.data);
        } else if (msg.type === 'perception') {
          setPerception(msg.data);
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      // Reconnect after 2s
      reconnectTimer.current = window.setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [token]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { state, perception, connected, sendMessage };
}

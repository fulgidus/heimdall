/**
 * WebSocket Manager for Real-time Dashboard Updates
 *
 * Features:
 * - Auto-reconnection with exponential backoff
 * - Event subscription/unsubscription
 * - Connection state tracking
 * - Heartbeat/ping-pong for keep-alive
 */

export const ConnectionState = {
  DISCONNECTED: 'Disconnected',
  CONNECTING: 'Connecting',
  CONNECTED: 'Connected',
  RECONNECTING: 'Reconnecting',
} as const;

export type ConnectionState = (typeof ConnectionState)[keyof typeof ConnectionState];

export interface WebSocketMessage {
  event: string;
  data: any;
  timestamp: string;
}

export type EventCallback = (data: any) => void;

interface WebSocketConfig {
  url: string;
  heartbeatInterval?: number;
  maxReconnectDelay?: number;
  initialReconnectDelay?: number;
  connectTimeout?: number;
}

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private reconnectDelay = 1000; // Start with 1 second
  private maxReconnectDelay: number;
  private initialReconnectDelay: number;
  private heartbeatInterval: number;
  private connectTimeout: number;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private connectTimer: NodeJS.Timeout | null = null;
  private state: ConnectionState = ConnectionState.DISCONNECTED;
  private subscribers: Map<string, Set<EventCallback>> = new Map();
  private stateChangeCallbacks: Set<(state: ConnectionState) => void> = new Set();
  private shouldReconnect = true;

  constructor(config: WebSocketConfig) {
    this.url = config.url;
    this.heartbeatInterval = config.heartbeatInterval ?? 30000; // 30 seconds
    this.maxReconnectDelay = config.maxReconnectDelay ?? 30000; // 30 seconds max
    this.initialReconnectDelay = config.initialReconnectDelay ?? 1000; // 1 second initial
    this.connectTimeout = config.connectTimeout ?? 5000; // 5 seconds to connect
  }

  /**
   * Connect to WebSocket server
   */
  public async connect(): Promise<void> {
    // Guard: If already connected or connecting, skip
    if (
      this.ws &&
      (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)
    ) {
      console.log('[WebSocket] Already connected or connecting');
      return;
    }

    // Guard: Clean up any stale websocket before reconnecting
    if (
      this.ws &&
      (this.ws.readyState === WebSocket.CLOSING || this.ws.readyState === WebSocket.CLOSED)
    ) {
      console.log('[WebSocket] Cleaning up closed connection before reconnecting');
      this.ws = null;
    }

    this.shouldReconnect = true;
    this.setState(ConnectionState.CONNECTING);

    return new Promise((resolve, reject) => {
      try {
        console.log(`[WebSocket] Connecting to ${this.url}`);
        this.ws = new WebSocket(this.url);

        // Set up connection timeout
        this.connectTimer = setTimeout(() => {
          console.warn('[WebSocket] Connection timeout');
          if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
            this.ws.close();
            reject(new Error('Connection timeout'));
          }
        }, this.connectTimeout);

        this.ws.onopen = () => {
          console.log('[WebSocket] Connected');
          if (this.connectTimer) {
            clearTimeout(this.connectTimer);
            this.connectTimer = null;
          }
          this.reconnectAttempts = 0;
          this.reconnectDelay = this.initialReconnectDelay;
          this.setState(ConnectionState.CONNECTED);
          this.startHeartbeat();
          resolve();
        };

        this.ws.onmessage = event => {
          this.handleMessage(event.data);
        };

        this.ws.onerror = error => {
          console.error('[WebSocket] Error:', error);
          if (this.connectTimer) {
            clearTimeout(this.connectTimer);
            this.connectTimer = null;
          }
          reject(error);
        };

        this.ws.onclose = event => {
          console.log('[WebSocket] Connection closed', event.code, event.reason);
          if (this.connectTimer) {
            clearTimeout(this.connectTimer);
            this.connectTimer = null;
          }
          this.stopHeartbeat();
          this.setState(ConnectionState.DISCONNECTED);

          if (this.shouldReconnect) {
            this.scheduleReconnect();
          }
        };
      } catch (error) {
        console.error('[WebSocket] Connection error:', error);
        if (this.connectTimer) {
          clearTimeout(this.connectTimer);
          this.connectTimer = null;
        }
        this.setState(ConnectionState.DISCONNECTED);
        reject(error);
      }
    });
  }

  /**
   * Disconnect from WebSocket server
   */
  public disconnect(): void {
    // Guard: Don't disconnect if already disconnected or never connected
    if (!this.ws || this.state === ConnectionState.DISCONNECTED) {
      console.log('[WebSocket] Already disconnected, skipping');
      return;
    }

    console.log('[WebSocket] Disconnecting');
    this.shouldReconnect = false;
    this.stopHeartbeat();
    this.clearReconnectTimer();

    if (this.ws) {
      // Only close if connection is in a closeable state
      if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
        this.ws.close();
      }
      this.ws = null;
    }

    this.setState(ConnectionState.DISCONNECTED);
  }

  /**
   * Subscribe to an event
   */
  public subscribe(event: string, callback: EventCallback): void {
    if (!this.subscribers.has(event)) {
      this.subscribers.set(event, new Set());
    }
    this.subscribers.get(event)!.add(callback);
    console.log(`[WebSocket] Subscribed to event: ${event}`);
  }

  /**
   * Unsubscribe from an event
   */
  public unsubscribe(event: string, callback: EventCallback): void {
    const callbacks = this.subscribers.get(event);
    if (callbacks) {
      callbacks.delete(callback);
      if (callbacks.size === 0) {
        this.subscribers.delete(event);
      }
      console.log(`[WebSocket] Unsubscribed from event: ${event}`);
    }
  }

  /**
   * Send a message to the server
   */
  public send(event: string, data: any): void {
    if (!this.isConnected()) {
      console.warn('[WebSocket] Cannot send message - not connected');
      return;
    }

    const message: WebSocketMessage = {
      event,
      data,
      timestamp: new Date().toISOString(),
    };

    this.ws!.send(JSON.stringify(message));
  }

  /**
   * Check if connected
   */
  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get current connection state
   */
  public getState(): ConnectionState {
    return this.state;
  }

  /**
   * Subscribe to connection state changes
   */
  public onStateChange(callback: (state: ConnectionState) => void): void {
    this.stateChangeCallbacks.add(callback);
  }

  /**
   * Unsubscribe from connection state changes
   */
  public offStateChange(callback: (state: ConnectionState) => void): void {
    this.stateChangeCallbacks.delete(callback);
  }

  /**
   * Handle incoming message
   */
  private handleMessage(data: string): void {
    try {
      const message: WebSocketMessage = JSON.parse(data);

      // Handle heartbeat response
      if (message.event === 'pong') {
        console.log('[WebSocket] Heartbeat acknowledged');
        return;
      }

      // Notify subscribers
      const callbacks = this.subscribers.get(message.event);
      if (callbacks) {
        callbacks.forEach(callback => {
          try {
            callback(message.data);
          } catch (error) {
            console.error(`[WebSocket] Error in callback for event ${message.event}:`, error);
          }
        });
      }
    } catch (error) {
      console.error('[WebSocket] Failed to parse message:', error);
    }
  }

  /**
   * Set connection state and notify listeners
   */
  private setState(state: ConnectionState): void {
    this.state = state;
    this.stateChangeCallbacks.forEach(callback => {
      try {
        callback(state);
      } catch (error) {
        console.error('[WebSocket] Error in state change callback:', error);
      }
    });
  }

  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected()) {
        this.send('ping', {});
      }
    }, this.heartbeatInterval);
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  private scheduleReconnect(): void {
    this.clearReconnectTimer();
    this.setState(ConnectionState.RECONNECTING);

    this.reconnectAttempts++;
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      this.maxReconnectDelay
    );

    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(error => {
        console.error('[WebSocket] Reconnection failed:', error);
      });
    }, delay);
  }

  /**
   * Clear reconnection timer
   */
  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}

/**
 * Create a WebSocket manager instance
 */
export function createWebSocketManager(url: string): WebSocketManager {
  return new WebSocketManager({
    url,
    heartbeatInterval: 30000,
    maxReconnectDelay: 30000,
    initialReconnectDelay: 1000,
  });
}

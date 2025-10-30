/**
 * WebSocket Context Provider
 * 
 * Provides a singleton WebSocket connection shared across all components.
 * Prevents race conditions by:
 * - Using a single manager instance
 * - Tracking component mount status
 * - Preventing state updates after unmount
 */

import React, { createContext, useContext, useEffect, useRef, useCallback, useState } from 'react';
import { WebSocketManager, ConnectionState, createWebSocketManager } from '@/lib/websocket';

interface WebSocketContextValue {
    manager: WebSocketManager | null;
    connectionState: ConnectionState;
    isConnected: boolean;
    connect: () => Promise<void>;
    disconnect: () => void;
    subscribe: (event: string, callback: (data: any) => void) => void;
    unsubscribe: (event: string, callback: (data: any) => void) => void;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

interface WebSocketProviderProps {
    children: React.ReactNode;
    url?: string;
    autoConnect?: boolean;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ 
    children, 
    url,
    autoConnect = true 
}) => {
    const managerRef = useRef<WebSocketManager | null>(null);
    const isMountedRef = useRef<boolean>(true);
    const connectingRef = useRef<boolean>(false);
    const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
    
    // Get WebSocket URL from props or construct from environment
    const getWebSocketUrl = useCallback(() => {
        if (url) return url;
        
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const hostname = window.location.hostname;
        const port = window.location.port || (protocol === 'wss' ? '443' : '80');
        return import.meta.env.VITE_SOCKET_URL || `${protocol}://${hostname}:${port}/ws`;
    }, [url]);

    // Initialize manager only once
    const initializeManager = useCallback(() => {
        if (managerRef.current) {
            return managerRef.current;
        }

        const wsUrl = getWebSocketUrl();
        console.log('[WebSocketContext] Initializing WebSocket manager with URL:', wsUrl);
        
        const manager = createWebSocketManager(wsUrl);
        
        // Subscribe to state changes with mount guard
        manager.onStateChange((state) => {
            if (isMountedRef.current) {
                setConnectionState(state);
            }
        });
        
        managerRef.current = manager;
        return manager;
    }, [getWebSocketUrl]);

    const connect = useCallback(async () => {
        // Prevent concurrent connection attempts
        if (connectingRef.current) {
            console.log('[WebSocketContext] Connection already in progress, skipping');
            return;
        }

        const manager = initializeManager();
        
        // Don't reconnect if already connected or connecting
        if (manager.isConnected() || manager.getState() === ConnectionState.CONNECTING) {
            console.log('[WebSocketContext] Already connected or connecting, skipping');
            return;
        }

        try {
            connectingRef.current = true;
            await manager.connect();
            console.log('[WebSocketContext] Connected successfully');
        } catch (error) {
            console.error('[WebSocketContext] Connection failed:', error);
        } finally {
            connectingRef.current = false;
        }
    }, [initializeManager]);

    const disconnect = useCallback(() => {
        if (managerRef.current) {
            console.log('[WebSocketContext] Disconnecting WebSocket');
            managerRef.current.disconnect();
        }
    }, []);

    const subscribe = useCallback((event: string, callback: (data: any) => void) => {
        const manager = initializeManager();
        
        // Wrap callback with mount guard to prevent updates after unmount
        const guardedCallback = (data: any) => {
            if (isMountedRef.current) {
                callback(data);
            }
        };
        
        manager.subscribe(event, guardedCallback);
    }, [initializeManager]);

    const unsubscribe = useCallback((event: string, callback: (data: any) => void) => {
        if (managerRef.current) {
            managerRef.current.unsubscribe(event, callback);
        }
    }, []);

    // Auto-connect on mount if enabled
    useEffect(() => {
        isMountedRef.current = true;

        if (autoConnect) {
            connect();
        }

        // Cleanup on unmount - disconnect only on final unmount
        return () => {
            isMountedRef.current = false;
            
            // Use a small delay to handle React StrictMode double-mounting
            setTimeout(() => {
                if (managerRef.current) {
                    console.log('[WebSocketContext] Component unmounted, disconnecting WebSocket');
                    managerRef.current.disconnect();
                    managerRef.current = null;
                }
            }, 100);
        };
    }, [autoConnect, connect]);

    const value: WebSocketContextValue = {
        manager: managerRef.current,
        connectionState,
        isConnected: managerRef.current?.isConnected() ?? false,
        connect,
        disconnect,
        subscribe,
        unsubscribe,
    };

    return (
        <WebSocketContext.Provider value={value}>
            {children}
        </WebSocketContext.Provider>
    );
};

/**
 * Hook to access WebSocket context
 */
export const useWebSocket = (): WebSocketContextValue => {
    const context = useContext(WebSocketContext);
    
    if (!context) {
        throw new Error('useWebSocket must be used within a WebSocketProvider');
    }
    
    return context;
};

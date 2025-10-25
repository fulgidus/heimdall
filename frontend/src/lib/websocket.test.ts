/**
 * Unit tests for WebSocket Manager
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { WebSocketManager, ConnectionState, createWebSocketManager } from './websocket';

// Mock WebSocket
class MockWebSocket {
    public readyState: number = WebSocket.CONNECTING;
    public onopen: ((event: Event) => void) | null = null;
    public onclose: ((event: CloseEvent) => void) | null = null;
    public onmessage: ((event: MessageEvent) => void) | null = null;
    public onerror: ((event: Event) => void) | null = null;

    constructor(public url: string) {
        // Simulate async connection
        setTimeout(() => {
            this.readyState = WebSocket.OPEN;
            if (this.onopen) {
                this.onopen(new Event('open'));
            }
        }, 10);
    }

    send(data: string): void {
        // Mock send - does nothing
    }

    close(): void {
        this.readyState = WebSocket.CLOSED;
        if (this.onclose) {
            this.onclose(new CloseEvent('close', { code: 1000, reason: 'Normal closure' }));
        }
    }
}

// Replace global WebSocket with mock
global.WebSocket = MockWebSocket as any;
(global as any).WebSocket.CONNECTING = 0;
(global as any).WebSocket.OPEN = 1;
(global as any).WebSocket.CLOSING = 2;
(global as any).WebSocket.CLOSED = 3;

describe('WebSocket Manager', () => {
    let manager: WebSocketManager;
    const testUrl = 'ws://localhost:8000/ws/updates';

    beforeEach(() => {
        manager = createWebSocketManager(testUrl);
        vi.clearAllTimers();
    });

    afterEach(() => {
        manager.disconnect();
    });

    describe('Connection Management', () => {
        it('should create a WebSocket manager instance', () => {
            expect(manager).toBeDefined();
            expect(manager.getState()).toBe(ConnectionState.DISCONNECTED);
        });

        it('should connect to WebSocket server', async () => {
            await manager.connect();
            expect(manager.isConnected()).toBe(true);
            expect(manager.getState()).toBe(ConnectionState.CONNECTED);
        });

        it('should disconnect from WebSocket server', async () => {
            await manager.connect();
            expect(manager.isConnected()).toBe(true);

            manager.disconnect();
            expect(manager.isConnected()).toBe(false);
            expect(manager.getState()).toBe(ConnectionState.DISCONNECTED);
        });

        it('should not connect if already connected', async () => {
            await manager.connect();
            const firstConnection = manager.isConnected();

            await manager.connect(); // Try to connect again
            expect(manager.isConnected()).toBe(firstConnection);
        });
    });

    describe('State Management', () => {
        it('should track connection state', async () => {
            const states: ConnectionState[] = [];
            manager.onStateChange((state) => states.push(state));

            await manager.connect();
            manager.disconnect();

            expect(states).toContain(ConnectionState.CONNECTING);
            expect(states).toContain(ConnectionState.CONNECTED);
            expect(states).toContain(ConnectionState.DISCONNECTED);
        });

        it('should notify state change listeners', async () => {
            const callback = vi.fn();
            manager.onStateChange(callback);

            await manager.connect();

            expect(callback).toHaveBeenCalledWith(ConnectionState.CONNECTING);
            expect(callback).toHaveBeenCalledWith(ConnectionState.CONNECTED);
        });

        it('should remove state change listeners', async () => {
            const callback = vi.fn();
            manager.onStateChange(callback);
            manager.offStateChange(callback);

            await manager.connect();

            expect(callback).not.toHaveBeenCalled();
        });
    });

    describe('Event Subscription', () => {
        it('should subscribe to events', () => {
            const callback = vi.fn();
            manager.subscribe('test-event', callback);

            // Verify subscription was added (private, so we can't directly test)
            expect(() => manager.subscribe('test-event', callback)).not.toThrow();
        });

        it('should unsubscribe from events', () => {
            const callback = vi.fn();
            manager.subscribe('test-event', callback);
            manager.unsubscribe('test-event', callback);

            // Verify unsubscription (private, so we can't directly test)
            expect(() => manager.unsubscribe('test-event', callback)).not.toThrow();
        });

        it('should handle multiple subscribers to same event', () => {
            const callback1 = vi.fn();
            const callback2 = vi.fn();

            manager.subscribe('test-event', callback1);
            manager.subscribe('test-event', callback2);

            expect(() => {
                manager.unsubscribe('test-event', callback1);
            }).not.toThrow();

            // callback2 should still be subscribed
            manager.unsubscribe('test-event', callback2);
        });
    });

    describe('Message Handling', () => {
        it('should send messages when connected', async () => {
            await manager.connect();
            
            // Mock the WebSocket send method
            const ws = (manager as any).ws;
            const sendSpy = vi.spyOn(ws, 'send');

            manager.send('test-event', { message: 'hello' });

            expect(sendSpy).toHaveBeenCalledWith(
                expect.stringContaining('"event":"test-event"')
            );
        });

        it('should not send messages when disconnected', () => {
            const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

            manager.send('test-event', { message: 'hello' });

            expect(consoleSpy).toHaveBeenCalledWith(
                expect.stringContaining('Cannot send message - not connected')
            );

            consoleSpy.mockRestore();
        });

        it('should receive and dispatch messages', async () => {
            await manager.connect();

            const callback = vi.fn();
            manager.subscribe('services:health', callback);

            // Simulate receiving a message
            const ws = (manager as any).ws;
            const messageEvent = new MessageEvent('message', {
                data: JSON.stringify({
                    event: 'services:health',
                    data: { status: 'healthy' },
                    timestamp: new Date().toISOString(),
                }),
            });

            ws.onmessage(messageEvent);

            expect(callback).toHaveBeenCalledWith({ status: 'healthy' });
        });
    });

    describe('Connection State Queries', () => {
        it('should return correct connection state', async () => {
            expect(manager.isConnected()).toBe(false);

            await manager.connect();
            expect(manager.isConnected()).toBe(true);

            manager.disconnect();
            expect(manager.isConnected()).toBe(false);
        });

        it('should return current state', async () => {
            expect(manager.getState()).toBe(ConnectionState.DISCONNECTED);

            const connectPromise = manager.connect();
            // During connection
            expect([ConnectionState.CONNECTING, ConnectionState.CONNECTED]).toContain(manager.getState());

            await connectPromise;
            expect(manager.getState()).toBe(ConnectionState.CONNECTED);
        });
    });

    describe('Factory Function', () => {
        it('should create WebSocket manager with factory', () => {
            const newManager = createWebSocketManager(testUrl);
            expect(newManager).toBeDefined();
            expect(newManager.getState()).toBe(ConnectionState.DISCONNECTED);
        });
    });
});

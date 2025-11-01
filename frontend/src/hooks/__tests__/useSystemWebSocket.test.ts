/**
 * Tests for useSystemWebSocket hook
 */

import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useSystemWebSocket } from '../useSystemWebSocket';
import { useSystemStore } from '@/store/systemStore';

// Mock the WebSocket context
vi.mock('@/contexts/WebSocketContext', () => ({
  useWebSocket: vi.fn(() => ({
    subscribe: vi.fn(() => vi.fn()),
    isConnected: true,
  })),
}));

// Mock the system store
vi.mock('@/store/systemStore', () => ({
  useSystemStore: vi.fn(() => ({
    updateServicesHealthFromWebSocket: vi.fn(),
  })),
}));

describe('useSystemWebSocket', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should subscribe to services:health event when connected', () => {
    const mockSubscribe = vi.fn(() => vi.fn());
    const mockUpdateServicesHealth = vi.fn();

    vi.mocked(useSystemStore).mockReturnValue({
      updateServicesHealthFromWebSocket: mockUpdateServicesHealth,
    } as any);

    const { useWebSocket } = require('@/contexts/WebSocketContext');
    vi.mocked(useWebSocket).mockReturnValue({
      subscribe: mockSubscribe,
      isConnected: true,
    });

    renderHook(() => useSystemWebSocket());

    expect(mockSubscribe).toHaveBeenCalledWith('services:health', expect.any(Function));
  });

  it('should not subscribe when not connected', () => {
    const mockSubscribe = vi.fn(() => vi.fn());

    const { useWebSocket } = require('@/contexts/WebSocketContext');
    vi.mocked(useWebSocket).mockReturnValue({
      subscribe: mockSubscribe,
      isConnected: false,
    });

    renderHook(() => useSystemWebSocket());

    expect(mockSubscribe).not.toHaveBeenCalled();
  });

  it('should update store when receiving health data', async () => {
    const mockUpdateServicesHealth = vi.fn();
    let capturedCallback: ((data: any) => void) | null = null;

    const mockSubscribe = vi.fn((event: string, callback: (data: any) => void) => {
      capturedCallback = callback;
      return vi.fn();
    });

    vi.mocked(useSystemStore).mockReturnValue({
      updateServicesHealthFromWebSocket: mockUpdateServicesHealth,
    } as any);

    const { useWebSocket } = require('@/contexts/WebSocketContext');
    vi.mocked(useWebSocket).mockReturnValue({
      subscribe: mockSubscribe,
      isConnected: true,
    });

    renderHook(() => useSystemWebSocket());

    // Simulate receiving WebSocket data
    const mockHealthData = {
      health_status: {
        backend: { status: 'healthy', service: 'backend' },
        training: { status: 'healthy', service: 'training' },
      },
    };

    if (capturedCallback) {
      capturedCallback(mockHealthData);
    }

    await waitFor(() => {
      expect(mockUpdateServicesHealth).toHaveBeenCalledWith(mockHealthData.health_status);
    });
  });

  it('should cleanup subscriptions on unmount', () => {
    const mockUnsubscribe = vi.fn();
    const mockSubscribe = vi.fn(() => mockUnsubscribe);

    const { useWebSocket } = require('@/contexts/WebSocketContext');
    vi.mocked(useWebSocket).mockReturnValue({
      subscribe: mockSubscribe,
      isConnected: true,
    });

    const { unmount } = renderHook(() => useSystemWebSocket());

    unmount();

    expect(mockUnsubscribe).toHaveBeenCalled();
  });

  it('should return isConnected status', () => {
    const { useWebSocket } = require('@/contexts/WebSocketContext');
    vi.mocked(useWebSocket).mockReturnValue({
      subscribe: vi.fn(() => vi.fn()),
      isConnected: true,
    });

    const { result } = renderHook(() => useSystemWebSocket());

    expect(result.current.isConnected).toBe(true);
  });
});

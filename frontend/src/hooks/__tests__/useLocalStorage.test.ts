import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useLocalStorage } from '../useLocalStorage';

describe('useLocalStorage Hook', () => {
    beforeEach(() => {
        localStorage.clear();
    });

    it('returns initial value when localStorage is empty', () => {
        const { result } = renderHook(() => useLocalStorage('test-key', 'initial-value'));

        expect(result.current[0]).toBe('initial-value');
    });

    it('returns stored value from localStorage', () => {
        localStorage.setItem('test-key', JSON.stringify('stored-value'));

        const { result } = renderHook(() => useLocalStorage('test-key', 'initial-value'));

        expect(result.current[0]).toBe('stored-value');
    });

    it('stores value in localStorage when setValue is called', () => {
        const { result } = renderHook(() => useLocalStorage('test-key', 'initial'));

        act(() => {
            result.current[1]('new-value');
        });

        expect(result.current[0]).toBe('new-value');
        expect(localStorage.getItem('test-key')).toBe(JSON.stringify('new-value'));
    });

    it('handles objects correctly', () => {
        const { result } = renderHook(() =>
            useLocalStorage('test-key', { name: 'Test', count: 0 })
        );

        act(() => {
            result.current[1]({ name: 'Updated', count: 5 });
        });

        expect(result.current[0]).toEqual({ name: 'Updated', count: 5 });
        expect(JSON.parse(localStorage.getItem('test-key')!)).toEqual({
            name: 'Updated',
            count: 5,
        });
    });

    it('handles arrays correctly', () => {
        const { result } = renderHook(() => useLocalStorage('test-key', [1, 2, 3]));

        act(() => {
            result.current[1]([4, 5, 6]);
        });

        expect(result.current[0]).toEqual([4, 5, 6]);
        expect(JSON.parse(localStorage.getItem('test-key')!)).toEqual([4, 5, 6]);
    });

    it('handles numbers correctly', () => {
        const { result } = renderHook(() => useLocalStorage('test-key', 42));

        act(() => {
            result.current[1](100);
        });

        expect(result.current[0]).toBe(100);
        expect(JSON.parse(localStorage.getItem('test-key')!)).toBe(100);
    });

    it('handles boolean values correctly', () => {
        const { result } = renderHook(() => useLocalStorage('test-key', false));

        act(() => {
            result.current[1](true);
        });

        expect(result.current[0]).toBe(true);
        expect(JSON.parse(localStorage.getItem('test-key')!)).toBe(true);
    });

    it('handles function updater pattern', () => {
        const { result } = renderHook(() => useLocalStorage('test-key', 10));

        act(() => {
            result.current[1]((prev) => prev + 5);
        });

        expect(result.current[0]).toBe(15);
        expect(JSON.parse(localStorage.getItem('test-key')!)).toBe(15);
    });

    it('handles function updater with objects', () => {
        const { result } = renderHook(() =>
            useLocalStorage('test-key', { count: 0, name: 'Test' })
        );

        act(() => {
            result.current[1]((prev) => ({ ...prev, count: prev.count + 1 }));
        });

        expect(result.current[0]).toEqual({ count: 1, name: 'Test' });
    });

    it('returns initial value when localStorage contains invalid JSON', () => {
        localStorage.setItem('test-key', 'invalid-json-{');

        const { result } = renderHook(() => useLocalStorage('test-key', 'fallback'));

        expect(result.current[0]).toBe('fallback');
    });

    it('gracefully handles localStorage errors during read', () => {
        // Mock localStorage.getItem to throw error
        const originalGetItem = Storage.prototype.getItem;
        Storage.prototype.getItem = () => {
            throw new Error('Storage read error');
        };

        const { result } = renderHook(() => useLocalStorage('test-key', 'default'));

        expect(result.current[0]).toBe('default');

        // Restore original method
        Storage.prototype.getItem = originalGetItem;
    });

    it('gracefully handles localStorage errors during write', () => {
        const { result } = renderHook(() => useLocalStorage('test-key', 'initial'));

        // Mock localStorage.setItem to throw error
        const originalSetItem = Storage.prototype.setItem;
        Storage.prototype.setItem = () => {
            throw new Error('Storage write error');
        };

        act(() => {
            result.current[1]('new-value'); // Should not crash
        });

        // Value is updated in state but not in storage
        expect(result.current[0]).toBe('new-value');

        // Restore original method
        Storage.prototype.setItem = originalSetItem;
    });

    it('uses different keys independently', () => {
        const { result: result1 } = renderHook(() => useLocalStorage('key1', 'value1'));
        const { result: result2 } = renderHook(() => useLocalStorage('key2', 'value2'));

        expect(result1.current[0]).toBe('value1');
        expect(result2.current[0]).toBe('value2');

        act(() => {
            result1.current[1]('updated1');
        });

        expect(result1.current[0]).toBe('updated1');
        expect(result2.current[0]).toBe('value2'); // Should not change
    });

    it('persists value across hook remounts', () => {
        const { result: result1, unmount } = renderHook(() =>
            useLocalStorage('test-key', 'initial')
        );

        act(() => {
            result1.current[1]('persisted-value');
        });

        unmount();

        const { result: result2 } = renderHook(() => useLocalStorage('test-key', 'initial'));

        expect(result2.current[0]).toBe('persisted-value');
    });
});

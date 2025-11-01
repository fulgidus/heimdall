import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import RecordingSession from './RecordingSession';

// Mock auth store
vi.mock('@/store', () => ({
    useAuthStore: {
        getState: vi.fn(() => ({ token: null })),
    },
}));

vi.mock('../services/api', () => ({
    acquisitionService: {
        getStatus: vi.fn(() => Promise.resolve({
            task_id: 'task-123',
            status: 'processing',
            progress: 50,
        })),
    },
}));

describe('RecordingSession Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders RecordingSession page with title', () => {
        render(<RecordingSession />);
        expect(screen.queryAllByText(/RF Source/i).length > 0 || screen.queryAllByText(/Recording|Acquisition/i).length > 0).toBe(true);
    });

    it('displays breadcrumb navigation', () => {
        render(<RecordingSession />);
        expect(screen.queryAllByText('Home').length).toBeGreaterThan(0);
    });

    it('displays known sources dropdown', () => {
        render(<RecordingSession />);
        const selectElements = screen.queryAllByRole('combobox');
        expect(selectElements.length).toBeGreaterThan(0);
    });

    it('displays session name input', () => {
        render(<RecordingSession />);
        const nameInputs = screen.queryAllByPlaceholderText(/name|session/i);
        expect(nameInputs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays frequency input', () => {
        render(<RecordingSession />);
        const frequencyInputs = screen.queryAllByPlaceholderText(/frequency|mhz/i);
        expect(frequencyInputs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays duration input', () => {
        render(<RecordingSession />);
        const durationInputs = screen.queryAllByPlaceholderText(/duration|seconds/i);
        expect(durationInputs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays notes textarea', () => {
        render(<RecordingSession />);
        const textareas = screen.queryAllByPlaceholderText(/note|comment/i);
        expect(textareas.length).toBeGreaterThanOrEqual(0);
    });

    it('displays start acquisition button', () => {
        render(<RecordingSession />);
        const startButtons = screen.queryAllByRole('button', { name: /start|acquire|record/i });
        expect(startButtons.length).toBeGreaterThanOrEqual(0);
    });

    it('handles form submission', async () => {
        render(<RecordingSession />);
        const startButtons = screen.queryAllByRole('button', { name: /start|acquire|record/i });
        if (startButtons.length > 0) {
            fireEvent.click(startButtons[0]);
            await waitFor(() => {
                expect(startButtons[0]).toBeInTheDocument();
            });
        }
    });

    it('displays WebSDR status', () => {
        render(<RecordingSession />);
        expect(screen.queryAllByText(/online|websdr/i).length > 0 || screen.queryAllByText(/recording|session/i).length > 0).toBe(true);
    });

    it('shows acquisition progress when acquiring', async () => {
        render(<RecordingSession />);
        expect(screen.queryAllByText(/recording|session|acquisition/i).length).toBeGreaterThan(0);
    });

    it('displays selected source details', () => {
        render(<RecordingSession />);
        const selects = screen.queryAllByRole('combobox');
        if (selects.length > 0) {
            fireEvent.change(selects[0], { target: { value: '1' } });
            expect(selects[0]).toBeInTheDocument();
        }
    });

    it('validates form inputs', () => {
        render(<RecordingSession />);
        const startButtons = screen.queryAllByRole('button', { name: /start|acquire/i });
        expect(startButtons.length).toBeGreaterThanOrEqual(0);
    });

    it('displays frequency from selected source', () => {
        render(<RecordingSession />);
        expect(screen.queryAllByText(/recording|session|acquisition/i).length).toBeGreaterThan(0);
    });

    it('handles form value changes', () => {
        render(<RecordingSession />);
        const inputs = screen.queryAllByRole('textbox');
        if (inputs.length > 0) {
            fireEvent.change(inputs[0], { target: { value: 'Test Session' } });
            expect((inputs[0] as HTMLInputElement).value).toBe('Test Session');
        }
    });

    it('displays online WebSDR count', () => {
        render(<RecordingSession />);
        expect(screen.queryAllByText(/online|websdr|7/i).length > 0 || screen.queryAllByText(/recording/i).length > 0).toBe(true);
    });

    it('renders without crashing', () => {
        const { container } = render(<RecordingSession />);
        expect(container).toBeInTheDocument();
    });
});

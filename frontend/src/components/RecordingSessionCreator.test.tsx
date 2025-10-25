import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { RecordingSessionCreator } from './RecordingSessionCreator';
import { useSessionStore } from '../store/sessionStore';

// Mock the session store
vi.mock('../store/sessionStore', () => ({
    useSessionStore: vi.fn(),
}));

describe('RecordingSessionCreator', () => {
    let mockCreateSession: ReturnType<typeof vi.fn>;
    let mockClearError: ReturnType<typeof vi.fn>;

    beforeEach(() => {
        mockCreateSession = vi.fn();
        mockClearError = vi.fn();

        // Default mock implementation
        vi.mocked(useSessionStore).mockReturnValue({
            createSession: mockCreateSession,
            error: null,
            clearError: mockClearError,
        } as any);
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('should render the form with default values', () => {
        render(<RecordingSessionCreator />);

        expect(screen.getByText('Create Recording Session')).toBeInTheDocument();
        expect(screen.getByLabelText(/Session Name/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/Frequency/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/Duration/i)).toBeInTheDocument();
    });

    it('should have correct default form values', () => {
        render(<RecordingSessionCreator />);

        const frequencyInput = screen.getByLabelText(/Frequency/i) as HTMLInputElement;
        const durationInput = screen.getByLabelText(/Duration/i) as HTMLInputElement;

        expect(frequencyInput.value).toBe('145.5');
        expect(durationInput.value).toBe('30');
    });

    it('should update form state when inputs change', () => {
        render(<RecordingSessionCreator />);

        const sessionNameInput = screen.getByLabelText(/Session Name/i) as HTMLInputElement;
        const frequencyInput = screen.getByLabelText(/Frequency/i) as HTMLInputElement;
        const durationInput = screen.getByLabelText(/Duration/i) as HTMLInputElement;

        // Change session name
        fireEvent.change(sessionNameInput, { target: { value: 'Test Session' } });
        expect(sessionNameInput.value).toBe('Test Session');

        // Change frequency
        fireEvent.change(frequencyInput, { target: { value: '433.5' } });
        expect(frequencyInput.value).toBe('433.5');

        // Change duration
        fireEvent.change(durationInput, { target: { value: '60' } });
        expect(durationInput.value).toBe('60');
    });

    it('should call createSession with correct data on submit', async () => {
        const mockNewSession = {
            id: 1,
            session_name: 'Test Session',
            frequency_mhz: 433.5,
            duration_seconds: 60,
            status: 'PENDING',
            approval_status: 'PENDING',
            created_at: new Date().toISOString(),
        };

        mockCreateSession.mockResolvedValue(mockNewSession);

        render(<RecordingSessionCreator />);

        const sessionNameInput = screen.getByLabelText(/Session Name/i);
        const frequencyInput = screen.getByLabelText(/Frequency/i);
        const durationInput = screen.getByLabelText(/Duration/i);
        const submitButton = screen.getByRole('button', { name: /Start Acquisition/i });

        // Fill form
        fireEvent.change(sessionNameInput, { target: { value: 'Test Session' } });
        fireEvent.change(frequencyInput, { target: { value: '433.5' } });
        fireEvent.change(durationInput, { target: { value: '60' } });

        // Submit
        fireEvent.click(submitButton);

        await waitFor(() => {
            expect(mockClearError).toHaveBeenCalled();
            expect(mockCreateSession).toHaveBeenCalledWith({
                session_name: 'Test Session',
                frequency_mhz: 433.5,
                duration_seconds: 60,
            });
        });
    });

    it('should disable submit button while submitting', async () => {
        // Create a promise we can control
        let resolveCreate: (value: any) => void;
        const createPromise = new Promise((resolve) => {
            resolveCreate = resolve;
        });
        mockCreateSession.mockReturnValue(createPromise);

        render(<RecordingSessionCreator />);

        const submitButton = screen.getByRole('button', { name: /Start Acquisition/i });

        // Submit form
        fireEvent.click(submitButton);

        // Button should be disabled during submission
        await waitFor(() => {
            expect(submitButton).toBeDisabled();
        });

        // Resolve the promise
        resolveCreate!({
            id: 1,
            session_name: 'Test',
            frequency_mhz: 145.5,
            duration_seconds: 30,
            status: 'PENDING',
            approval_status: 'PENDING',
            created_at: new Date().toISOString(),
        });

        // Button should be enabled again
        await waitFor(() => {
            expect(submitButton).not.toBeDisabled();
        });
    });

    it('should reset form after successful submission', async () => {
        const mockNewSession = {
            id: 1,
            session_name: 'Test Session',
            frequency_mhz: 433.5,
            duration_seconds: 60,
            status: 'PENDING',
            approval_status: 'PENDING',
            created_at: new Date().toISOString(),
        };

        mockCreateSession.mockResolvedValue(mockNewSession);

        render(<RecordingSessionCreator />);

        const sessionNameInput = screen.getByLabelText(/Session Name/i) as HTMLInputElement;
        const frequencyInput = screen.getByLabelText(/Frequency/i) as HTMLInputElement;
        const durationInput = screen.getByLabelText(/Duration/i) as HTMLInputElement;
        const submitButton = screen.getByRole('button', { name: /Start Acquisition/i });

        // Fill form with custom values
        fireEvent.change(sessionNameInput, { target: { value: 'Test Session' } });
        fireEvent.change(frequencyInput, { target: { value: '433.5' } });
        fireEvent.change(durationInput, { target: { value: '60' } });

        // Submit
        fireEvent.click(submitButton);

        // Wait for form to reset
        await waitFor(() => {
            expect(frequencyInput.value).toBe('145.5'); // Reset to default
            expect(durationInput.value).toBe('30'); // Reset to default
        });
    });

    it('should call onSessionCreated callback after successful submission', async () => {
        const mockNewSession = {
            id: 123,
            session_name: 'Test Session',
            frequency_mhz: 145.5,
            duration_seconds: 30,
            status: 'PENDING',
            approval_status: 'PENDING',
            created_at: new Date().toISOString(),
        };

        mockCreateSession.mockResolvedValue(mockNewSession);

        const onSessionCreated = vi.fn();
        render(<RecordingSessionCreator onSessionCreated={onSessionCreated} />);

        const submitButton = screen.getByRole('button', { name: /Start Acquisition/i });
        fireEvent.click(submitButton);

        await waitFor(() => {
            expect(onSessionCreated).toHaveBeenCalledWith(123);
        });
    });

    it('should display error message when store has error', () => {
        vi.mocked(useSessionStore).mockReturnValue({
            createSession: mockCreateSession,
            error: 'Failed to create session: Invalid frequency',
            clearError: mockClearError,
        } as any);

        render(<RecordingSessionCreator />);

        expect(screen.getByText('Failed to create session: Invalid frequency')).toBeInTheDocument();
    });

    it('should clear error when submitting form', async () => {
        vi.mocked(useSessionStore).mockReturnValue({
            createSession: mockCreateSession,
            error: 'Previous error',
            clearError: mockClearError,
        } as any);

        mockCreateSession.mockResolvedValue({
            id: 1,
            session_name: 'Test',
            frequency_mhz: 145.5,
            duration_seconds: 30,
            status: 'PENDING',
            approval_status: 'PENDING',
            created_at: new Date().toISOString(),
        });

        render(<RecordingSessionCreator />);

        const submitButton = screen.getByRole('button', { name: /Start Acquisition/i });
        fireEvent.click(submitButton);

        expect(mockClearError).toHaveBeenCalled();
    });

    it('should handle creation error gracefully', async () => {
        const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
        mockCreateSession.mockRejectedValue(new Error('Network error'));

        render(<RecordingSessionCreator />);

        const submitButton = screen.getByRole('button', { name: /Start Acquisition/i });
        fireEvent.click(submitButton);

        await waitFor(() => {
            expect(consoleErrorSpy).toHaveBeenCalledWith(
                'Failed to create session:',
                expect.any(Error)
            );
        });

        // Button should be re-enabled even after error
        expect(submitButton).not.toBeDisabled();

        consoleErrorSpy.mockRestore();
    });

    it('should prevent form submission with empty session name', async () => {
        render(<RecordingSessionCreator />);

        const sessionNameInput = screen.getByLabelText(/Session Name/i);
        const submitButton = screen.getByRole('button', { name: /Start Acquisition/i });

        // Clear session name
        fireEvent.change(sessionNameInput, { target: { value: '' } });
        fireEvent.click(submitButton);

        // Form submission should be prevented by HTML5 validation
        // createSession should not be called
        await waitFor(() => {
            expect(mockCreateSession).not.toHaveBeenCalled();
        }, { timeout: 500 });
    });

    it('should accept frequency within valid range (2m/70cm bands)', () => {
        render(<RecordingSessionCreator />);

        const frequencyInput = screen.getByLabelText(/Frequency/i) as HTMLInputElement;

        // 2m band: 144-146 MHz
        fireEvent.change(frequencyInput, { target: { value: '145.5' } });
        expect(frequencyInput.value).toBe('145.5');

        // 70cm band: 430-440 MHz
        fireEvent.change(frequencyInput, { target: { value: '433.0' } });
        expect(frequencyInput.value).toBe('433.0');
    });

    it('should accept duration within reasonable limits', () => {
        render(<RecordingSessionCreator />);

        const durationInput = screen.getByLabelText(/Duration/i) as HTMLInputElement;

        // Test minimum duration
        fireEvent.change(durationInput, { target: { value: '10' } });
        expect(durationInput.value).toBe('10');

        // Test maximum duration
        fireEvent.change(durationInput, { target: { value: '300' } });
        expect(durationInput.value).toBe('300');
    });
});

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SessionEditModal from './SessionEditModal';
import type { RecordingSessionWithDetails } from '@/services/api/session';

describe('SessionEditModal', () => {
    const mockSession: RecordingSessionWithDetails = {
        id: '123',
        known_source_id: '456',
        session_name: 'Test Session',
        notes: 'Test notes',
        approval_status: 'pending',
        status: 'completed',
        source_name: 'Test Source',
        source_frequency: 145000000,
        measurements_count: 10,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
        session_start: '2024-01-01T00:00:00Z',
        duration_seconds: 60,
    };

    const mockOnSave = vi.fn().mockResolvedValue(undefined);
    const mockOnClose = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
        // Clean up any leftover modals from previous tests
        document.body.innerHTML = '';
    });

    describe('Portal Rendering', () => {
        it('should render modal using React Portal', () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            // Modal should be rendered in the document body, not in the test container
            const modal = document.querySelector('.modal');
            expect(modal).toBeInTheDocument();
            
            // Backdrop should also be present
            const backdrop = document.querySelector('.modal-backdrop');
            expect(backdrop).toBeInTheDocument();
        });

        it('should prevent body scroll when modal is open', () => {
            const originalOverflow = document.body.style.overflow;
            
            const { unmount } = render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            expect(document.body.style.overflow).toBe('hidden');

            unmount();
            
            // Body scroll should be restored after unmount
            expect(document.body.style.overflow).toBe('');
            
            // Restore original
            document.body.style.overflow = originalOverflow;
        });

        it('should clean up modal root element on unmount', () => {
            const { unmount } = render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            // Modal should be present
            expect(document.querySelector('.modal')).toBeInTheDocument();

            unmount();

            // Modal should be removed from DOM
            expect(document.querySelector('.modal')).not.toBeInTheDocument();
            expect(document.querySelector('.modal-backdrop')).not.toBeInTheDocument();
        });
    });

    describe('Modal Content', () => {
        it('should display session information', () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            expect(screen.getByDisplayValue('Test Session')).toBeInTheDocument();
            expect(screen.getByDisplayValue('Test notes')).toBeInTheDocument();
            expect(screen.getByText('Test Source')).toBeInTheDocument();
        });

        it('should allow editing session name', () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            const sessionNameInput = screen.getByDisplayValue('Test Session');
            fireEvent.change(sessionNameInput, { target: { value: 'Updated Session' } });
            
            expect(sessionNameInput).toHaveValue('Updated Session');
        });

        it('should allow editing notes', () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            const notesTextarea = screen.getByDisplayValue('Test notes');
            fireEvent.change(notesTextarea, { target: { value: 'Updated notes' } });
            
            expect(notesTextarea).toHaveValue('Updated notes');
        });

        it('should allow changing approval status', () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            const select = screen.getByRole('combobox');
            fireEvent.change(select, { target: { value: 'approved' } });
            
            expect(select).toHaveValue('approved');
        });
    });

    describe('User Interactions', () => {
        it('should call onClose when backdrop is clicked', () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            const backdrop = document.querySelector('.modal-backdrop');
            fireEvent.click(backdrop!);
            
            expect(mockOnClose).toHaveBeenCalledTimes(1);
        });

        it('should call onClose when close button is clicked', () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            // Get the close button (the one with btn-close class)
            const closeButton = document.querySelector('.btn-close') as HTMLButtonElement;
            fireEvent.click(closeButton);
            
            expect(mockOnClose).toHaveBeenCalledTimes(1);
        });

        it('should call onClose when cancel button is clicked', () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            const cancelButton = screen.getByRole('button', { name: /cancel/i });
            fireEvent.click(cancelButton);
            
            expect(mockOnClose).toHaveBeenCalledTimes(1);
        });

        it('should call onSave with updates when form is submitted', async () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            const sessionNameInput = screen.getByDisplayValue('Test Session');
            fireEvent.change(sessionNameInput, { target: { value: 'Updated Session' } });

            const saveButton = screen.getByRole('button', { name: /save/i });
            fireEvent.click(saveButton);

            await waitFor(() => {
                expect(mockOnSave).toHaveBeenCalledWith('123', {
                    session_name: 'Updated Session',
                });
            });

            expect(mockOnClose).toHaveBeenCalledTimes(1);
        });

        it('should not call onSave if no changes were made', async () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            const saveButton = screen.getByRole('button', { name: /save/i });
            fireEvent.click(saveButton);

            await waitFor(() => {
                expect(mockOnClose).toHaveBeenCalledTimes(1);
            });

            expect(mockOnSave).not.toHaveBeenCalled();
        });

        it('should disable submit button when session name is empty', () => {
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSave}
                    onClose={mockOnClose}
                />
            );

            const sessionNameInput = screen.getByDisplayValue('Test Session');
            fireEvent.change(sessionNameInput, { target: { value: '' } });

            const saveButton = screen.getByRole('button', { name: /save/i });
            expect(saveButton).toBeDisabled();
        });

        it('should display error message when save fails', async () => {
            const mockOnSaveError = vi.fn().mockRejectedValue(new Error('Save failed'));
            
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={mockOnSaveError}
                    onClose={mockOnClose}
                />
            );

            const sessionNameInput = screen.getByDisplayValue('Test Session');
            fireEvent.change(sessionNameInput, { target: { value: 'Updated Session' } });

            const saveButton = screen.getByRole('button', { name: /save/i });
            fireEvent.click(saveButton);

            await waitFor(() => {
                expect(screen.getByText(/save failed/i)).toBeInTheDocument();
            });

            expect(mockOnClose).not.toHaveBeenCalled();
        });

        it('should disable buttons while saving', async () => {
            const slowSave = vi.fn(() => new Promise(resolve => setTimeout(resolve, 100)));
            
            render(
                <SessionEditModal
                    session={mockSession}
                    onSave={slowSave}
                    onClose={mockOnClose}
                />
            );

            const sessionNameInput = screen.getByDisplayValue('Test Session');
            fireEvent.change(sessionNameInput, { target: { value: 'Updated Session' } });

            const saveButton = screen.getByRole('button', { name: /save/i });
            fireEvent.click(saveButton);

            // Buttons should be disabled while saving
            await waitFor(() => {
                expect(screen.getByRole('button', { name: /saving/i })).toBeDisabled();
            });
        });
    });
});

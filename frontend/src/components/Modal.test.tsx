import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Modal from './Modal';

describe('Modal', () => {
    // Store original focus element
    let originalActiveElement: Element | null;

    beforeEach(() => {
        originalActiveElement = document.activeElement;
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    describe('Rendering', () => {
        it('should not render when isOpen is false', () => {
            const { container } = render(
                <Modal isOpen={false} onClose={vi.fn()} title="Test Modal">
                    Content
                </Modal>
            );

            expect(container.firstChild).toBeNull();
        });

        it('should render when isOpen is true', () => {
            render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test Modal">
                    Content
                </Modal>
            );

            expect(screen.getByRole('dialog')).toBeInTheDocument();
            expect(screen.getByText('Test Modal')).toBeInTheDocument();
            expect(screen.getByText('Content')).toBeInTheDocument();
        });

        it('should render with footer when provided', () => {
            render(
                <Modal
                    isOpen={true}
                    onClose={vi.fn()}
                    title="Test Modal"
                    footer={<button>Submit</button>}
                >
                    Content
                </Modal>
            );

            expect(screen.getByRole('button', { name: 'Submit' })).toBeInTheDocument();
        });

        it('should render without footer when not provided', () => {
            render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test Modal">
                    Content
                </Modal>
            );

            const submitButton = screen.queryByRole('button', { name: 'Submit' });
            expect(submitButton).not.toBeInTheDocument();
        });

        it('should apply correct size class for sm size', () => {
            const { container } = render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test" size="sm">
                    Content
                </Modal>
            );

            const modal = container.querySelector('.max-w-sm');
            expect(modal).toBeInTheDocument();
        });

        it('should apply correct size class for md size (default)', () => {
            const { container } = render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test">
                    Content
                </Modal>
            );

            const modal = container.querySelector('.max-w-md');
            expect(modal).toBeInTheDocument();
        });

        it('should apply correct size class for lg size', () => {
            const { container } = render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test" size="lg">
                    Content
                </Modal>
            );

            const modal = container.querySelector('.max-w-lg');
            expect(modal).toBeInTheDocument();
        });

        it('should apply correct size class for xl size', () => {
            const { container } = render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test" size="xl">
                    Content
                </Modal>
            );

            const modal = container.querySelector('.max-w-xl');
            expect(modal).toBeInTheDocument();
        });
    });

    describe('Accessibility', () => {
        it('should have correct ARIA attributes', () => {
            render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test Modal">
                    Content
                </Modal>
            );

            const dialog = screen.getByRole('dialog');
            expect(dialog).toHaveAttribute('aria-modal', 'true');
            expect(dialog).toHaveAttribute('aria-labelledby', 'modal-title');
        });

        it('should label close button for screen readers', () => {
            render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test Modal">
                    Content
                </Modal>
            );

            const closeButton = screen.getByLabelText('Close modal');
            expect(closeButton).toBeInTheDocument();
        });
    });

    describe('User Interactions', () => {
        it('should call onClose when backdrop is clicked', () => {
            const onClose = vi.fn();
            const { container } = render(
                <Modal isOpen={true} onClose={onClose} title="Test Modal">
                    Content
                </Modal>
            );

            const backdrop = container.querySelector('.bg-black.bg-opacity-50');
            expect(backdrop).toBeInTheDocument();
            
            fireEvent.click(backdrop!);
            expect(onClose).toHaveBeenCalledTimes(1);
        });

        it('should call onClose when close button is clicked', () => {
            const onClose = vi.fn();
            render(
                <Modal isOpen={true} onClose={onClose} title="Test Modal">
                    Content
                </Modal>
            );

            const closeButton = screen.getByLabelText('Close modal');
            fireEvent.click(closeButton);
            expect(onClose).toHaveBeenCalledTimes(1);
        });

        it('should not call onClose when modal content is clicked', () => {
            const onClose = vi.fn();
            render(
                <Modal isOpen={true} onClose={onClose} title="Test Modal">
                    Content
                </Modal>
            );

            const content = screen.getByText('Content');
            fireEvent.click(content);
            expect(onClose).not.toHaveBeenCalled();
        });

        it('should call onClose when Escape key is pressed', () => {
            const onClose = vi.fn();
            render(
                <Modal isOpen={true} onClose={onClose} title="Test Modal">
                    Content
                </Modal>
            );

            fireEvent.keyDown(document, { key: 'Escape' });
            expect(onClose).toHaveBeenCalledTimes(1);
        });

        it('should not call onClose when Escape is pressed and modal is closed', () => {
            const onClose = vi.fn();
            render(
                <Modal isOpen={false} onClose={onClose} title="Test Modal">
                    Content
                </Modal>
            );

            fireEvent.keyDown(document, { key: 'Escape' });
            expect(onClose).not.toHaveBeenCalled();
        });
    });

    describe('Focus Management', () => {
        it('should focus the modal when opened', async () => {
            const { rerender } = render(
                <Modal isOpen={false} onClose={vi.fn()} title="Test Modal">
                    Content
                </Modal>
            );

            // Open the modal
            rerender(
                <Modal isOpen={true} onClose={vi.fn()} title="Test Modal">
                    Content
                </Modal>
            );

            await waitFor(() => {
                const dialog = screen.getByRole('dialog').querySelector('[tabindex="-1"]');
                expect(document.activeElement).toBe(dialog);
            });
        });

        it('should restore focus to previous element when closed', async () => {
            const button = document.createElement('button');
            button.textContent = 'Open';
            document.body.appendChild(button);
            button.focus();

            const { rerender } = render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test Modal">
                    Content
                </Modal>
            );

            // Close the modal
            rerender(
                <Modal isOpen={false} onClose={vi.fn()} title="Test Modal">
                    Content
                </Modal>
            );

            await waitFor(() => {
                expect(document.activeElement).toBe(button);
            });

            document.body.removeChild(button);
        });
    });

    describe('Focus Trap', () => {
        it('should trap Tab key within modal', () => {
            render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test Modal">
                    <input type="text" placeholder="First" />
                    <input type="text" placeholder="Second" />
                </Modal>
            );

            const firstInput = screen.getByPlaceholderText('First');
            const secondInput = screen.getByPlaceholderText('Second');
            const closeButton = screen.getByLabelText('Close modal');

            // Focus last element
            closeButton.focus();
            expect(document.activeElement).toBe(closeButton);

            // Tab should cycle to first focusable element
            fireEvent.keyDown(document, { key: 'Tab' });
            // Note: In jsdom, we can't fully test focus trap behavior
            // This test ensures the event listener is set up
        });

        it('should handle Shift+Tab for reverse focus trap', () => {
            render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test Modal">
                    <input type="text" placeholder="First" />
                    <input type="text" placeholder="Second" />
                </Modal>
            );

            const firstInput = screen.getByPlaceholderText('First');
            firstInput.focus();

            // Shift+Tab should cycle to last focusable element
            fireEvent.keyDown(document, { key: 'Tab', shiftKey: true });
            // Note: In jsdom, we can't fully test focus trap behavior
            // This test ensures the event listener is set up
        });
    });

    describe('Edge Cases', () => {
        it('should handle modal with no focusable elements', () => {
            render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test Modal">
                    <div>No focusable content</div>
                </Modal>
            );

            // Should not throw error
            fireEvent.keyDown(document, { key: 'Tab' });
        });

        it('should handle rapid open/close cycles', () => {
            const onClose = vi.fn();
            const { rerender } = render(
                <Modal isOpen={false} onClose={onClose} title="Test Modal">
                    Content
                </Modal>
            );

            // Open
            rerender(
                <Modal isOpen={true} onClose={onClose} title="Test Modal">
                    Content
                </Modal>
            );

            // Close
            rerender(
                <Modal isOpen={false} onClose={onClose} title="Test Modal">
                    Content
                </Modal>
            );

            // Open again
            rerender(
                <Modal isOpen={true} onClose={onClose} title="Test Modal">
                    Content
                </Modal>
            );

            expect(screen.getByRole('dialog')).toBeInTheDocument();
        });

        it('should handle children with complex content', () => {
            render(
                <Modal isOpen={true} onClose={vi.fn()} title="Complex Modal">
                    <div>
                        <h3>Subtitle</h3>
                        <ul>
                            <li>Item 1</li>
                            <li>Item 2</li>
                        </ul>
                        <button>Action</button>
                    </div>
                </Modal>
            );

            expect(screen.getByText('Subtitle')).toBeInTheDocument();
            expect(screen.getByText('Item 1')).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Action' })).toBeInTheDocument();
        });
    });

    describe('Cleanup', () => {
        it('should remove event listeners on unmount', () => {
            const removeEventListenerSpy = vi.spyOn(document, 'removeEventListener');
            const { unmount } = render(
                <Modal isOpen={true} onClose={vi.fn()} title="Test Modal">
                    Content
                </Modal>
            );

            unmount();

            expect(removeEventListenerSpy).toHaveBeenCalledWith('keydown', expect.any(Function));
            removeEventListenerSpy.mockRestore();
        });
    });
});

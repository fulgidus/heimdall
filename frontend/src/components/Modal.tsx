import React, { useEffect, useRef } from 'react';
import classNames from 'classnames';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

const Modal: React.FC<ModalProps> = ({ isOpen, onClose, title, children, footer, size = 'md' }) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (isOpen) {
      // Store the element that had focus before the modal opened
      previousFocusRef.current = document.activeElement as HTMLElement;

      // Focus the modal
      modalRef.current?.focus();
    } else {
      // Return focus to the previously focused element
      previousFocusRef.current?.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }

      // Focus trap: keep focus within modal
      if (e.key === 'Tab' && isOpen && modalRef.current) {
        const focusableElements = modalRef.current.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const firstElement = focusableElements[0] as HTMLElement;
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

        if (e.shiftKey) {
          // Shift+Tab: if at first element, move to last
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement?.focus();
          }
        } else {
          // Tab: if at last element, move to first
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement?.focus();
          }
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Modal */}
      <div
        className="fixed inset-0 flex items-center justify-center z-50 p-4"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
      >
        <div
          ref={modalRef}
          className={classNames(
            'bg-oxford-blue rounded-lg shadow-2xl w-full',
            'border border-neon-blue border-opacity-20',
            'animate-slideIn',
            sizeClasses[size]
          )}
          onClick={e => e.stopPropagation()}
          tabIndex={-1}
        >
          {/* Header */}
          <div className="px-6 py-4 border-b border-neon-blue border-opacity-20 flex items-center justify-between">
            <h2 id="modal-title" className="text-xl font-bold text-white">
              {title}
            </h2>
            <button
              onClick={onClose}
              className="p-1 hover:bg-sea-green hover:bg-opacity-20 rounded transition-colors text-french-gray"
              aria-label="Close modal"
            >
              ✕
            </button>
          </div>

          {/* Content */}
          <div className="px-6 py-4 max-h-96 overflow-y-auto">{children}</div>

          {/* Footer */}
          {footer && (
            <div className="px-6 py-4 border-t border-neon-blue border-opacity-20 flex justify-end gap-3">
              {footer}
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default Modal;

import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import classNames from 'classnames';
import { usePortal } from '@/hooks/usePortal';

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
  
  // Use bulletproof portal hook (prevents removeChild errors)
  const portalTarget = usePortal(isOpen);

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

  // Don't render if not open or portal not ready
  if (!isOpen || !portalTarget) return null;

  const sizeClasses = {
    sm: 'modal-sm',
    md: '',
    lg: 'modal-lg',
    xl: 'modal-xl',
  };

  const modalContent = (
    <>
      {/* Bootstrap Modal Backdrop */}
      <div
        className="modal-backdrop fade show"
        onClick={onClose}
        aria-hidden="true"
        style={{ zIndex: 1040 }}
      />

      {/* Bootstrap Modal */}
      <div
        className="modal fade show d-block"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        style={{ zIndex: 1050 }}
        tabIndex={-1}
      >
        <div
          className={classNames('modal-dialog modal-dialog-centered modal-dialog-scrollable', sizeClasses[size])}
          onClick={e => e.stopPropagation()}
        >
          <div ref={modalRef} className="modal-content" tabIndex={-1}>
            {/* Header */}
            <div className="modal-header">
              <h5 id="modal-title" className="modal-title">
                {title}
              </h5>
              <button
                type="button"
                className="btn-close"
                onClick={onClose}
                aria-label="Close"
              />
            </div>

            {/* Content */}
            <div className="modal-body">{children}</div>

            {/* Footer */}
            {footer && (
              <div className="modal-footer">
                {footer}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );

  // Render modal through portal to prevent removeChild errors
  return createPortal(modalContent, portalTarget);
};

export default Modal;

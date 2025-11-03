import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { AVAILABLE_WIDGETS } from '@/types/widgets';
import { useWidgetStore } from '@/store/widgetStore';

interface WidgetPickerProps {
  show: boolean;
  onClose: () => void;
}

export const WidgetPicker: React.FC<WidgetPickerProps> = ({ show, onClose }) => {
  const { addWidget } = useWidgetStore();
  const modalRootRef = useRef<HTMLDivElement | null>(null);
  const isMountedRef = useRef(false);

  // Initialize portal container once
  if (!modalRootRef.current) {
    modalRootRef.current = document.createElement('div');
  }

  // Mount and unmount the modal root element
  useEffect(() => {
    const modalRoot = modalRootRef.current;
    if (!modalRoot) return;

    if (show && !isMountedRef.current) {
      document.body.appendChild(modalRoot);
      isMountedRef.current = true;
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    }

    if (!show && isMountedRef.current) {
      // Restore body scroll
      document.body.style.overflow = '';
      // Use timeout to allow React to finish rendering before removing
      const timeoutId = setTimeout(() => {
        if (modalRoot.parentNode === document.body) {
          document.body.removeChild(modalRoot);
          isMountedRef.current = false;
        }
      }, 0);
      
      return () => clearTimeout(timeoutId);
    }
  }, [show]);

  const handleAddWidget = (widgetType: string) => {
    addWidget(widgetType as any);
    onClose();
  };

  if (!show || !modalRootRef.current) return null;

  return createPortal(
    <>
      <div className="modal show d-block" tabIndex={-1}>
        <div className="modal-dialog modal-dialog-centered">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">Add Widget</h5>
              <button type="button" className="btn-close" onClick={onClose} />
            </div>
            <div className="modal-body">
              <div className="row g-3">
                {AVAILABLE_WIDGETS.map(widget => (
                  <div key={widget.type} className="col-12">
                    <div
                      className="card cursor-pointer hover-shadow"
                      onClick={() => handleAddWidget(widget.type)}
                      style={{ cursor: 'pointer' }}
                    >
                      <div className="card-body">
                        <div className="d-flex align-items-start gap-3">
                          <div className="flex-shrink-0">
                            <i className={`ph ${widget.icon} fs-1 text-primary`} />
                          </div>
                          <div className="flex-grow-1">
                            <h6 className="mb-1">{widget.title}</h6>
                            <p className="text-muted small mb-0">{widget.description}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" onClick={onClose}>
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
      <div className="modal-backdrop fade show" onClick={onClose} />
    </>,
    modalRootRef.current
  );
};

import React from 'react';
import { AVAILABLE_WIDGETS } from '@/types/widgets';
import { useWidgetStore } from '@/store/widgetStore';

interface WidgetPickerProps {
    show: boolean;
    onClose: () => void;
}

export const WidgetPicker: React.FC<WidgetPickerProps> = ({ show, onClose }) => {
    const { addWidget } = useWidgetStore();

    const handleAddWidget = (widgetType: string) => {
        addWidget(widgetType as any);
        onClose();
    };

    if (!show) return null;

    return (
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
                                {AVAILABLE_WIDGETS.map((widget) => (
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
                                                        <p className="text-muted small mb-0">
                                                            {widget.description}
                                                        </p>
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
        </>
    );
};

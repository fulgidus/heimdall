import React from 'react';
import type { WidgetConfig } from '@/types/widgets';
import { SIZE_TO_COLUMNS } from '@/types/widgets';
import { useWidgetStore } from '@/store/widgetStore';

interface WidgetContainerProps {
    widget: WidgetConfig;
    children: React.ReactNode;
}

export const WidgetContainer: React.FC<WidgetContainerProps> = ({ widget, children }) => {
    const { removeWidget } = useWidgetStore();
    const colClass = `col-12 col-md-${SIZE_TO_COLUMNS[widget.size]}`;

    return (
        <div className={colClass}>
            <div className="card h-100">
                <div className="card-header d-flex align-items-center justify-content-between">
                    <h5 className="mb-0">{widget.title}</h5>
                    <button
                        className="btn btn-sm btn-link-danger p-0"
                        onClick={() => removeWidget(widget.id)}
                        aria-label="Remove widget"
                        title="Remove widget"
                    >
                        <i className="ph ph-x" />
                    </button>
                </div>
                <div className="card-body">
                    {children}
                </div>
            </div>
        </div>
    );
};

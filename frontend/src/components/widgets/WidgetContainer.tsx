import React, { useState } from 'react';
import type { WidgetConfig } from '@/types/widgets';
import { SIZE_TO_COLUMNS, HEIGHT_TO_CLASS } from '@/types/widgets';
import { useWidgetStore } from '@/store/widgetStore';

interface WidgetContainerProps {
    widget: WidgetConfig;
    children: React.ReactNode;
}

export const WidgetContainer: React.FC<WidgetContainerProps> = ({ widget, children }) => {
    const { removeWidget, updateWidget } = useWidgetStore();
    const [showSizeMenu, setShowSizeMenu] = useState(false);
    
    // Use custom width if set, otherwise fallback to size-based columns
    const columns = widget.width ?? SIZE_TO_COLUMNS[widget.size];
    const colClass = `col-12 col-md-${columns}`;
    
    // Apply height class if set
    const heightClass = widget.height ? HEIGHT_TO_CLASS[widget.height] : '';

    const handleWidthChange = (newWidth: number) => {
        updateWidget(widget.id, { width: newWidth });
        setShowSizeMenu(false);
    };

    const handleHeightChange = (newHeight: WidgetConfig['height']) => {
        updateWidget(widget.id, { height: newHeight });
        setShowSizeMenu(false);
    };

    return (
        <div className={colClass}>
            <div className={`card h-100 ${heightClass}`}>
                <div className="card-header d-flex align-items-center justify-content-between">
                    <h5 className="mb-0">{widget.title}</h5>
                    <div className="d-flex gap-1">
                        <div className="position-relative">
                            <button
                                className="btn btn-sm btn-link-secondary p-0 px-1"
                                onClick={() => setShowSizeMenu(!showSizeMenu)}
                                aria-label="Resize widget"
                                title="Resize widget"
                            >
                                <i className="ph ph-arrows-out" />
                            </button>
                            {showSizeMenu && (
                                <>
                                    <div 
                                        className="position-fixed top-0 start-0 w-100 h-100" 
                                        style={{ zIndex: 1040 }}
                                        onClick={() => setShowSizeMenu(false)}
                                    />
                                    <div 
                                        className="position-absolute end-0 bg-white border rounded shadow-sm p-2"
                                        style={{ minWidth: '200px', zIndex: 1050 }}
                                    >
                                        <div className="mb-2">
                                            <small className="text-muted d-block mb-1">Width</small>
                                            <div className="btn-group btn-group-sm w-100" role="group">
                                                <button
                                                    className={`btn ${columns === 4 ? 'btn-primary' : 'btn-outline-secondary'}`}
                                                    onClick={() => handleWidthChange(4)}
                                                    title="Small width (4 columns)"
                                                >
                                                    S
                                                </button>
                                                <button
                                                    className={`btn ${columns === 6 ? 'btn-primary' : 'btn-outline-secondary'}`}
                                                    onClick={() => handleWidthChange(6)}
                                                    title="Medium width (6 columns)"
                                                >
                                                    M
                                                </button>
                                                <button
                                                    className={`btn ${columns === 8 ? 'btn-primary' : 'btn-outline-secondary'}`}
                                                    onClick={() => handleWidthChange(8)}
                                                    title="Large width (8 columns)"
                                                >
                                                    L
                                                </button>
                                                <button
                                                    className={`btn ${columns === 12 ? 'btn-primary' : 'btn-outline-secondary'}`}
                                                    onClick={() => handleWidthChange(12)}
                                                    title="Extra large width (12 columns)"
                                                >
                                                    XL
                                                </button>
                                            </div>
                                        </div>
                                        <div>
                                            <small className="text-muted d-block mb-1">Height</small>
                                            <div className="d-flex flex-column gap-1">
                                                <button
                                                    className={`btn btn-sm ${!widget.height || widget.height === 'auto' ? 'btn-primary' : 'btn-outline-secondary'} text-start`}
                                                    onClick={() => handleHeightChange('auto')}
                                                >
                                                    Auto
                                                </button>
                                                <button
                                                    className={`btn btn-sm ${widget.height === 'small' ? 'btn-primary' : 'btn-outline-secondary'} text-start`}
                                                    onClick={() => handleHeightChange('small')}
                                                >
                                                    Small
                                                </button>
                                                <button
                                                    className={`btn btn-sm ${widget.height === 'medium' ? 'btn-primary' : 'btn-outline-secondary'} text-start`}
                                                    onClick={() => handleHeightChange('medium')}
                                                >
                                                    Medium
                                                </button>
                                                <button
                                                    className={`btn btn-sm ${widget.height === 'large' ? 'btn-primary' : 'btn-outline-secondary'} text-start`}
                                                    onClick={() => handleHeightChange('large')}
                                                >
                                                    Large
                                                </button>
                                                <button
                                                    className={`btn btn-sm ${widget.height === 'xlarge' ? 'btn-primary' : 'btn-outline-secondary'} text-start`}
                                                    onClick={() => handleHeightChange('xlarge')}
                                                >
                                                    X-Large
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                        <button
                            className="btn btn-sm btn-link-danger p-0 px-1"
                            onClick={() => removeWidget(widget.id)}
                            aria-label="Remove widget"
                            title="Remove widget"
                        >
                            <i className="ph ph-x" />
                        </button>
                    </div>
                </div>
                <div className="card-body">
                    {children}
                </div>
            </div>
        </div>
    );
};

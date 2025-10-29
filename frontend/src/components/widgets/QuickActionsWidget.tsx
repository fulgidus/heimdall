import React from 'react';
import { Link } from 'react-router-dom';

interface QuickActionsWidgetProps {
    widgetId: string;
}

export const QuickActionsWidget: React.FC<QuickActionsWidgetProps> = () => {
    return (
        <div className="widget-content">
            <div className="d-grid gap-2">
                <Link to="/sessions/new" className="btn btn-primary">
                    <i className="ph ph-plus-circle me-2" />
                    New Recording Session
                </Link>
                
                <Link to="/sessions" className="btn btn-outline-primary">
                    <i className="ph ph-folder-open me-2" />
                    View Sessions
                </Link>
                
                <Link to="/localization" className="btn btn-outline-primary">
                    <i className="ph ph-map-pin me-2" />
                    Localization Map
                </Link>
                
                <Link to="/websdrs" className="btn btn-outline-secondary">
                    <i className="ph ph-radio-button me-2" />
                    WebSDR Status
                </Link>
                
                <Link to="/analytics" className="btn btn-outline-secondary">
                    <i className="ph ph-chart-line me-2" />
                    Analytics
                </Link>
            </div>
        </div>
    );
};

import React from 'react';
import classNames from 'classnames';
import './ChartCard.css';

export interface ChartCardProps {
    title: string;
    description?: string;
    children: React.ReactNode;
    className?: string;
    actions?: React.ReactNode;
    loading?: boolean;
    error?: string;
}

const ChartCard: React.FC<ChartCardProps> = ({
    title,
    description,
    children,
    className,
    actions,
    loading = false,
    error,
}) => {
    return (
        <div className={classNames('card chart-card', className)}>
            <div className="card-header">
                <div className="chart-card-header-content">
                    <div>
                        <h5 className="card-title mb-0">{title}</h5>
                        {description && (
                            <p className="text-muted small mb-0 mt-1">{description}</p>
                        )}
                    </div>
                    {actions && <div className="chart-card-actions">{actions}</div>}
                </div>
            </div>
            <div className="card-body">
                {loading && (
                    <div className="chart-loading">
                        <div className="spinner-border text-primary" role="status">
                            <span className="visually-hidden">Loading chart...</span>
                        </div>
                    </div>
                )}
                {error && (
                    <div className="alert alert-danger mb-0">
                        <i className="ph ph-warning me-2"></i>
                        {error}
                    </div>
                )}
                {!loading && !error && (
                    <div className="chart-container">{children}</div>
                )}
            </div>
        </div>
    );
};

export default ChartCard;

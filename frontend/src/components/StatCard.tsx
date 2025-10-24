import React from 'react';
import classNames from 'classnames';
import './StatCard.css';

export interface StatCardProps {
    title: string;
    value: string | number;
    icon?: string; // Phosphor icon class (e.g., "ph-activity")
    variant?: 'primary' | 'success' | 'warning' | 'danger' | 'info';
    trend?: {
        value: number;
        direction: 'up' | 'down';
        label?: string;
    };
    subtitle?: string;
    className?: string;
}

const StatCard: React.FC<StatCardProps> = ({
    title,
    value,
    icon,
    variant = 'primary',
    trend,
    subtitle,
    className,
}) => {
    const variantClasses = {
        primary: 'stat-card-primary',
        success: 'stat-card-success',
        warning: 'stat-card-warning',
        danger: 'stat-card-danger',
        info: 'stat-card-info',
    };

    const iconColors = {
        primary: 'text-primary',
        success: 'text-success',
        warning: 'text-warning',
        danger: 'text-danger',
        info: 'text-info',
    };

    return (
        <div className={classNames('card stat-card', variantClasses[variant], className)}>
            <div className="card-body">
                <div className="stat-card-content">
                    <div className="stat-card-info">
                        <h6 className="stat-card-title text-muted mb-2">{title}</h6>
                        <h3 className="stat-card-value mb-0">{value}</h3>
                        {subtitle && (
                            <p className="stat-card-subtitle text-muted mb-0 mt-1">
                                {subtitle}
                            </p>
                        )}
                        {trend && (
                            <div className="stat-card-trend mt-2">
                                <span
                                    className={classNames('stat-trend', {
                                        'stat-trend-up': trend.direction === 'up',
                                        'stat-trend-down': trend.direction === 'down',
                                    })}
                                >
                                    <i
                                        className={classNames('ph', {
                                            'ph-trend-up': trend.direction === 'up',
                                            'ph-trend-down': trend.direction === 'down',
                                        })}
                                    ></i>
                                    {Math.abs(trend.value)}%
                                </span>
                                {trend.label && (
                                    <span className="text-muted ms-2">{trend.label}</span>
                                )}
                            </div>
                        )}
                    </div>
                    {icon && (
                        <div className="stat-card-icon">
                            <i className={classNames('ph', icon, iconColors[variant])}></i>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default StatCard;

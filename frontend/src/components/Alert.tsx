import React from 'react';
import classNames from 'classnames';
import { AlertCircle, CheckCircle, AlertTriangle, Info, X } from 'lucide-react';
import './Alert.css';

interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: 'info' | 'success' | 'warning' | 'error';
    title?: string;
    message: string;
    closeable?: boolean;
    onClose?: () => void;
}

const Alert = React.forwardRef<HTMLDivElement, AlertProps>(
    ({ variant = 'info', title, message, closeable = false, onClose, className, ...props }, ref) => {
        const [isVisible, setIsVisible] = React.useState(true);

        const variantConfig = {
            info: {
                icon: Info,
                role: 'status',
            },
            success: {
                icon: CheckCircle,
                role: 'status',
            },
            warning: {
                icon: AlertTriangle,
                role: 'alert',
            },
            error: {
                icon: AlertCircle,
                role: 'alert',
            },
        };

        const config = variantConfig[variant];
        const Icon = config.icon;

        const handleClose = () => {
            setIsVisible(false);
            onClose?.();
        };

        if (!isVisible) return null;

        return (
            <div
                ref={ref}
                className={classNames(
                    'alert',
                    `alert-${variant}`,
                    className
                )}
                role={config.role}
                aria-live={variant === 'error' || variant === 'warning' ? 'assertive' : 'polite'}
                aria-atomic="true"
                {...props}
            >
                <Icon className="alert-icon" size={20} aria-hidden="true" />
                <div className="alert-content">
                    {title && <h3 className="alert-title">{title}</h3>}
                    <p className="alert-message">{message}</p>
                </div>
                {closeable && (
                    <button
                        onClick={handleClose}
                        className="alert-close-button"
                        aria-label={`Close ${variant} alert: ${title || message}`}
                    >
                        <X size={18} aria-hidden="true" />
                    </button>
                )}
            </div>
        );
    }
);

Alert.displayName = 'Alert';

export default Alert;

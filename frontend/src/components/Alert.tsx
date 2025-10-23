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
            },
            success: {
                icon: CheckCircle,
            },
            warning: {
                icon: AlertTriangle,
            },
            error: {
                icon: AlertCircle,
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
                {...props}
            >
                <Icon className="alert-icon" size={20} />
                <div className="alert-content">
                    {title && <h3 className="alert-title">{title}</h3>}
                    <p className="alert-message">{message}</p>
                </div>
                {closeable && (
                    <button
                        onClick={handleClose}
                        className="alert-close-button"
                    >
                        <X size={18} />
                    </button>
                )}
            </div>
        );
    }
);

Alert.displayName = 'Alert';

export default Alert;

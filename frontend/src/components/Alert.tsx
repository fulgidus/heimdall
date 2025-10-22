import React from 'react';
import classNames from 'classnames';
import { AlertCircle, CheckCircle, AlertTriangle, Info, X } from 'lucide-react';

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
                bg: 'bg-neon-blue bg-opacity-10',
                border: 'border-neon-blue',
                text: 'text-neon-blue',
                icon: Info,
            },
            success: {
                bg: 'bg-light-green bg-opacity-10',
                border: 'border-light-green',
                text: 'text-light-green',
                icon: CheckCircle,
            },
            warning: {
                bg: 'bg-yellow-500 bg-opacity-10',
                border: 'border-yellow-500',
                text: 'text-yellow-400',
                icon: AlertTriangle,
            },
            error: {
                bg: 'bg-red-500 bg-opacity-10',
                border: 'border-red-500',
                text: 'text-red-400',
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
                    'p-4 rounded-lg border flex items-start gap-4',
                    config.bg,
                    config.border,
                    className
                )}
                {...props}
            >
                <Icon className={classNames(config.text, 'flex-shrink-0 mt-0.5')} size={20} />
                <div className="flex-1 min-w-0">
                    {title && <h3 className={classNames('font-semibold mb-1', config.text)}>{title}</h3>}
                    <p className="text-french-gray text-sm">{message}</p>
                </div>
                {closeable && (
                    <button
                        onClick={handleClose}
                        className={classNames(
                            'flex-shrink-0 p-1 hover:bg-opacity-20 rounded transition-colors',
                            config.text
                        )}
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

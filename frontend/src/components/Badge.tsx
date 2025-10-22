import React from 'react';
import classNames from 'classnames';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
    variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'warning';
    size?: 'sm' | 'md' | 'lg';
    children: React.ReactNode;
}

const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
    ({ variant = 'primary', size = 'md', className, children, ...props }, ref) => {
        const variantClasses = {
            primary: 'bg-neon-blue bg-opacity-20 text-neon-blue border border-neon-blue border-opacity-50',
            secondary: 'bg-sea-green bg-opacity-20 text-light-green border border-sea-green border-opacity-50',
            success: 'bg-light-green bg-opacity-20 text-light-green border border-light-green border-opacity-50',
            danger: 'bg-red-500 bg-opacity-20 text-red-300 border border-red-500 border-opacity-50',
            warning: 'bg-yellow-500 bg-opacity-20 text-yellow-300 border border-yellow-500 border-opacity-50',
        };

        const sizeClasses = {
            sm: 'px-2 py-0.5 text-xs font-medium',
            md: 'px-3 py-1 text-sm font-medium',
            lg: 'px-4 py-1.5 text-base font-semibold',
        };

        return (
            <span
                ref={ref}
                className={classNames(
                    'inline-flex items-center rounded-full',
                    variantClasses[variant],
                    sizeClasses[size],
                    className
                )}
                {...props}
            >
                {children}
            </span>
        );
    }
);

Badge.displayName = 'Badge';

export default Badge;

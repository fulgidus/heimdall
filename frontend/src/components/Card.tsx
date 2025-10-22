import React from 'react';
import classNames from 'classnames';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'bordered' | 'elevated';
    children: React.ReactNode;
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
    ({ variant = 'default', className, children, ...props }, ref) => {
        const variantClasses = {
            default: 'bg-gradient-to-br from-oxford-blue to-sea-green border border-neon-blue border-opacity-20',
            bordered: 'bg-oxford-blue border-2 border-neon-blue',
            elevated: 'bg-oxford-blue border border-french-gray border-opacity-30 shadow-xl',
        };

        return (
            <div
                ref={ref}
                className={classNames(
                    'rounded-lg p-6 transition-all duration-200 backdrop-blur-sm',
                    variantClasses[variant],
                    className
                )}
                {...props}
            >
                {children}
            </div>
        );
    }
);

Card.displayName = 'Card';

export default Card;

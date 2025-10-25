import React from 'react';
import classNames from 'classnames';
import './Card.css';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'bordered' | 'elevated';
    children: React.ReactNode;
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
    ({ variant = 'default', className, children, ...props }, ref) => {
        return (
            <div
                ref={ref}
                className={classNames(
                    'card',
                    `card-${variant}`,
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

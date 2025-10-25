import React from 'react';
import classNames from 'classnames';
import './Input.css';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
    helperText?: string;
    icon?: React.ReactNode;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
    ({ label, error, helperText, icon, className, ...props }, ref) => {
        return (
            <div className="input-wrapper">
                {label && (
                    <label className="input-label">
                        {label}
                    </label>
                )}
                <div className="input-container">
                    {icon && (
                        <div className="input-icon">
                            {icon}
                        </div>
                    )}
                    <input
                        ref={ref}
                        className={classNames(
                            'input-field',
                            { 'input-field-with-icon': !!icon },
                            { 'input-field-error': !!error },
                            className
                        )}
                        {...props}
                    />
                </div>
                {error && (
                    <p className="input-error-text">{error}</p>
                )}
                {helperText && !error && (
                    <p className="input-helper-text">{helperText}</p>
                )}
            </div>
        );
    }
);

Input.displayName = 'Input';

export default Input;

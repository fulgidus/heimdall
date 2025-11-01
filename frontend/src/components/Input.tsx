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
  ({ label, error, helperText, icon, className, id, required, ...props }, ref) => {
    // Generate a unique ID if not provided
    const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
    const errorId = error ? `${inputId}-error` : undefined;
    const helperId = helperText ? `${inputId}-helper` : undefined;

    return (
      <div className="input-wrapper">
        {label && (
          <label htmlFor={inputId} className="input-label">
            {label}
            {required && (
              <span className="text-red-500 ml-1" aria-label="required">
                *
              </span>
            )}
          </label>
        )}
        <div className="input-container">
          {icon && (
            <div className="input-icon" aria-hidden="true">
              {icon}
            </div>
          )}
          <input
            ref={ref}
            id={inputId}
            className={classNames(
              'input-field',
              { 'input-field-with-icon': !!icon },
              { 'input-field-error': !!error },
              className
            )}
            aria-invalid={error ? 'true' : 'false'}
            aria-describedby={error ? errorId : helperId}
            aria-required={required}
            {...props}
          />
        </div>
        {error && (
          <p id={errorId} className="input-error-text" role="alert">
            {error}
          </p>
        )}
        {helperText && !error && (
          <p id={helperId} className="input-helper-text">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;

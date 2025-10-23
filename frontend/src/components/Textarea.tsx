import React from 'react';
import classNames from 'classnames';
import './Textarea.css';

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
    label?: string;
    error?: string;
    helperText?: string;
    fullWidth?: boolean;
    rows?: number;
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
    ({ label, error, helperText, fullWidth = false, rows = 4, className, ...props }, ref) => {
        const textareaId = props.id || `textarea-${Math.random().toString(36).substr(2, 9)}`;

        return (
            <div className={classNames('textarea-group', { 'w-100': fullWidth })}>
                {label && (
                    <label htmlFor={textareaId} className="form-label">
                        {label}
                        {props.required && <span className="text-danger ms-1">*</span>}
                    </label>
                )}
                <textarea
                    ref={ref}
                    id={textareaId}
                    rows={rows}
                    className={classNames(
                        'form-control',
                        {
                            'is-invalid': error,
                        },
                        className
                    )}
                    {...props}
                />
                {error && <div className="invalid-feedback d-block">{error}</div>}
                {helperText && !error && (
                    <div className="form-text">{helperText}</div>
                )}
            </div>
        );
    }
);

Textarea.displayName = 'Textarea';

export default Textarea;

import React from 'react';
import classNames from 'classnames';
import './Select.css';

export interface SelectOption {
  value: string | number;
  label: string;
  disabled?: boolean;
}

interface SelectProps extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  label?: string;
  options: SelectOption[];
  error?: string;
  helperText?: string;
  size?: 'sm' | 'md' | 'lg';
  fullWidth?: boolean;
}

const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  (
    { label, options, error, helperText, size = 'md', fullWidth = false, className, ...props },
    ref
  ) => {
    const selectId = props.id || `select-${Math.random().toString(36).substr(2, 9)}`;

    return (
      <div className={classNames('select-group', { 'w-100': fullWidth })}>
        {label && (
          <label htmlFor={selectId} className="form-label">
            {label}
            {props.required && <span className="text-danger ms-1">*</span>}
          </label>
        )}
        <select
          ref={ref}
          id={selectId}
          className={classNames(
            'form-select',
            `form-select-${size}`,
            {
              'is-invalid': error,
            },
            className
          )}
          {...props}
        >
          {options.map(option => (
            <option key={option.value} value={option.value} disabled={option.disabled}>
              {option.label}
            </option>
          ))}
        </select>
        {error && <div className="invalid-feedback d-block">{error}</div>}
        {helperText && !error && <div className="form-text">{helperText}</div>}
      </div>
    );
  }
);

Select.displayName = 'Select';

export default Select;

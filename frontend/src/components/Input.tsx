import React from 'react';
import classNames from 'classnames';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label?: string;
    error?: string;
    helperText?: string;
    icon?: React.ReactNode;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
    ({ label, error, helperText, icon, className, ...props }, ref) => {
        return (
            <div className="w-full">
                {label && (
                    <label className="block text-sm font-medium text-french-gray mb-2">
                        {label}
                    </label>
                )}
                <div className="relative">
                    {icon && (
                        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-neon-blue">
                            {icon}
                        </div>
                    )}
                    <input
                        ref={ref}
                        className={classNames(
                            'w-full px-4 py-2.5 rounded-lg',
                            'bg-oxford-blue border border-neon-blue border-opacity-30',
                            'text-white placeholder-french-gray placeholder-opacity-50',
                            'focus:outline-none focus:ring-2 focus:ring-neon-blue focus:border-transparent',
                            'focus:bg-opacity-50 transition-all duration-200',
                            { 'pl-10': !!icon },
                            { 'border-red-500 focus:ring-red-500': !!error },
                            className
                        )}
                        {...props}
                    />
                </div>
                {error && (
                    <p className="text-red-400 text-sm mt-1">{error}</p>
                )}
                {helperText && !error && (
                    <p className="text-french-gray text-sm mt-1 opacity-75">{helperText}</p>
                )}
            </div>
        );
    }
);

Input.displayName = 'Input';

export default Input;

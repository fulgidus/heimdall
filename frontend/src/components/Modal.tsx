import React from 'react';
import classNames from 'classnames';

interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    children: React.ReactNode;
    footer?: React.ReactNode;
    size?: 'sm' | 'md' | 'lg' | 'xl';
}

const Modal: React.FC<ModalProps> = ({ isOpen, onClose, title, children, footer, size = 'md' }) => {
    if (!isOpen) return null;

    const sizeClasses = {
        sm: 'max-w-sm',
        md: 'max-w-md',
        lg: 'max-w-lg',
        xl: 'max-w-xl',
    };

    return (
        <>
            {/* Backdrop */}
            <div
                className="fixed inset-0 bg-black bg-opacity-50 z-40 transition-opacity"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
                <div
                    className={classNames(
                        'bg-oxford-blue rounded-lg shadow-2xl w-full',
                        'border border-neon-blue border-opacity-20',
                        'animate-slideIn',
                        sizeClasses[size]
                    )}
                    onClick={(e) => e.stopPropagation()}
                >
                    {/* Header */}
                    <div className="px-6 py-4 border-b border-neon-blue border-opacity-20 flex items-center justify-between">
                        <h2 className="text-xl font-bold text-white">{title}</h2>
                        <button
                            onClick={onClose}
                            className="p-1 hover:bg-sea-green hover:bg-opacity-20 rounded transition-colors text-french-gray"
                        >
                            ✕
                        </button>
                    </div>

                    {/* Content */}
                    <div className="px-6 py-4 max-h-96 overflow-y-auto">
                        {children}
                    </div>

                    {/* Footer */}
                    {footer && (
                        <div className="px-6 py-4 border-t border-neon-blue border-opacity-20 flex justify-end gap-3">
                            {footer}
                        </div>
                    )}
                </div>
            </div>
        </>
    );
};

export default Modal;

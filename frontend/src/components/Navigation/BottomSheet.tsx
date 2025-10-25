import React, { useEffect, useState } from 'react';
import { X } from 'lucide-react';

interface BottomSheetProps {
    isOpen: boolean;
    onClose: () => void;
    title?: string;
    children: React.ReactNode;
    maxHeight?: string;
}

const BottomSheet: React.FC<BottomSheetProps> = ({ 
    isOpen, 
    onClose, 
    title, 
    children,
    maxHeight = '80vh'
}) => {
    const [startY, setStartY] = useState<number>(0);
    const [currentY, setCurrentY] = useState<number>(0);
    const [isDragging, setIsDragging] = useState(false);

    useEffect(() => {
        if (isOpen) {
            // Prevent body scroll when sheet is open
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
        return () => {
            document.body.style.overflow = '';
        };
    }, [isOpen]);

    const handleTouchStart = (e: React.TouchEvent) => {
        setStartY(e.touches[0].clientY);
        setIsDragging(true);
    };

    const handleTouchMove = (e: React.TouchEvent) => {
        if (!isDragging) return;
        setCurrentY(e.touches[0].clientY);
    };

    const handleTouchEnd = () => {
        if (!isDragging) return;
        setIsDragging(false);
        
        // If dragged down more than 100px, close the sheet
        if (currentY - startY > 100) {
            onClose();
        }
        
        setStartY(0);
        setCurrentY(0);
    };

    const translateY = isDragging && currentY > startY ? currentY - startY : 0;

    return (
        <>
            {/* Backdrop */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 z-40"
                    style={{ 
                        backdropFilter: 'blur(4px)',
                        transition: 'opacity 0.3s ease-in-out',
                        opacity: isOpen ? 1 : 0
                    }}
                    onClick={onClose}
                />
            )}

            {/* Bottom Sheet */}
            <div
                className={`fixed bottom-0 left-0 right-0 bg-oxford-blue rounded-t-2xl shadow-2xl z-50 transition-transform duration-300 ease-in-out ${
                    isOpen ? 'translate-y-0' : 'translate-y-full'
                }`}
                style={{
                    backgroundColor: 'var(--oxford-blue)',
                    maxHeight,
                    transform: isOpen ? `translateY(${translateY}px)` : 'translateY(100%)',
                    paddingBottom: 'env(safe-area-inset-bottom, 0px)', // iOS safe area
                }}
            >
                {/* Drag Handle */}
                <div
                    className="flex justify-center pt-3 pb-2 cursor-grab active:cursor-grabbing"
                    onTouchStart={handleTouchStart}
                    onTouchMove={handleTouchMove}
                    onTouchEnd={handleTouchEnd}
                >
                    <div
                        className="w-12 h-1 rounded-full"
                        style={{ backgroundColor: 'var(--french-gray)' }}
                    />
                </div>

                {/* Header */}
                {title && (
                    <div className="flex items-center justify-between px-4 py-3 border-b border-neon-blue border-opacity-20">
                        <h3 className="text-lg font-semibold text-white">{title}</h3>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-sea-green hover:bg-opacity-20 rounded-lg transition-colors touch-target"
                            aria-label="Close"
                        >
                            <X size={20} className="text-neon-blue" />
                        </button>
                    </div>
                )}

                {/* Content */}
                <div 
                    className="overflow-y-auto p-4"
                    style={{
                        maxHeight: `calc(${maxHeight} - 120px)`
                    }}
                >
                    {children}
                </div>
            </div>
        </>
    );
};

export default BottomSheet;

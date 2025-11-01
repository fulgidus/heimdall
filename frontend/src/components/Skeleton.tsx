import React from 'react';
import './Skeleton.css';

interface SkeletonProps {
  width?: string;
  height?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  className?: string;
}

/**
 * Skeleton loader component for displaying loading placeholders
 */
const Skeleton: React.FC<SkeletonProps> = ({
  width = '100%',
  height = '20px',
  variant = 'rectangular',
  className = '',
}) => {
  const getVariantClass = () => {
    switch (variant) {
      case 'text':
        return 'skeleton-text';
      case 'circular':
        return 'skeleton-circular';
      case 'rectangular':
      default:
        return 'skeleton-rectangular';
    }
  };

  return (
    <div
      className={`skeleton ${getVariantClass()} ${className}`}
      style={{ width, height }}
      aria-busy="true"
      aria-live="polite"
    />
  );
};

export default Skeleton;

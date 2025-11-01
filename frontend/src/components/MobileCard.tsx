import React from 'react';

interface MobileCardProps {
  children: React.ReactNode;
  className?: string;
}

const MobileCard: React.FC<MobileCardProps> = ({ children, className = '' }) => {
  return (
    <div
      className={`bg-sea-green bg-opacity-10 border border-neon-blue border-opacity-20 rounded-lg p-4 shadow-sm ${className}`}
      style={{
        backgroundColor: 'rgba(6, 167, 125, 0.1)',
      }}
    >
      {children}
    </div>
  );
};

export default MobileCard;

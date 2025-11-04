/**
 * InlineEditText Component
 * 
 * A reusable component for inline text editing with Enter/Escape key handling.
 * Click text to edit, press Enter to save, or Escape to cancel.
 */

import { useState, useRef, useEffect } from 'react';

interface InlineEditTextProps {
  value: string;
  onSave: (newValue: string) => Promise<void>;
  className?: string;
  inputClassName?: string;
  placeholder?: string;
  maxLength?: number;
  disabled?: boolean;
}

export const InlineEditText = ({
  value,
  onSave,
  className = '',
  inputClassName = '',
  placeholder = 'Enter text...',
  maxLength = 100,
  disabled = false,
}: InlineEditTextProps) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(value);
  const [isSaving, setIsSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Sync editValue with prop value when not editing
  useEffect(() => {
    if (!isEditing) {
      setEditValue(value);
    }
  }, [value, isEditing]);

  // Focus and select text when entering edit mode
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleClick = () => {
    if (!disabled && !isEditing) {
      setIsEditing(true);
    }
  };

  const handleSave = async () => {
    const trimmedValue = editValue.trim();

    // Don't save if empty or unchanged
    if (!trimmedValue || trimmedValue === value) {
      setIsEditing(false);
      setEditValue(value); // Reset to original
      return;
    }

    setIsSaving(true);
    try {
      await onSave(trimmedValue);
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to save:', error);
      // Keep editing mode open on error so user can retry
      setEditValue(trimmedValue);
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancel = () => {
    setEditValue(value); // Reset to original
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      handleCancel();
    }
  };

  const handleBlur = () => {
    // Save on blur (clicking outside the input)
    if (!isSaving) {
      handleSave();
    }
  };

  if (isEditing) {
    return (
      <input
        ref={inputRef}
        type="text"
        value={editValue}
        onChange={(e) => setEditValue(e.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        className={`form-control ${inputClassName}`}
        placeholder={placeholder}
        maxLength={maxLength}
        disabled={isSaving}
      />
    );
  }

  return (
    <span
      onClick={handleClick}
      className={`${className} ${disabled ? '' : 'cursor-pointer hover:text-primary'}`}
      role="button"
      tabIndex={disabled ? -1 : 0}
      onKeyDown={(e) => {
        if (!disabled && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          handleClick();
        }
      }}
      title={disabled ? '' : 'Click to edit'}
      style={{ cursor: disabled ? 'default' : 'pointer' }}
    >
      {value || placeholder}
    </span>
  );
};

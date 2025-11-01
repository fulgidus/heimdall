import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import BottomSheet from './BottomSheet';

describe('BottomSheet Component', () => {
  it('renders when open', () => {
    const onClose = vi.fn();
    render(
      <BottomSheet isOpen={true} onClose={onClose} title="Test Sheet">
        <div>Test Content</div>
      </BottomSheet>
    );

    expect(screen.getByText('Test Sheet')).toBeInTheDocument();
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });

  it('renders without title', () => {
    const onClose = vi.fn();
    render(
      <BottomSheet isOpen={true} onClose={onClose}>
        <div>Test Content</div>
      </BottomSheet>
    );

    expect(screen.getByText('Test Content')).toBeInTheDocument();
    expect(screen.queryByRole('heading')).not.toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = vi.fn();
    render(
      <BottomSheet isOpen={true} onClose={onClose} title="Test">
        <div>Content</div>
      </BottomSheet>
    );

    const closeButton = screen.getByLabelText('Close');
    fireEvent.click(closeButton);

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when backdrop is clicked', () => {
    const onClose = vi.fn();
    const { container } = render(
      <BottomSheet isOpen={true} onClose={onClose}>
        <div>Content</div>
      </BottomSheet>
    );

    const backdrop = container.querySelector('.bg-black.bg-opacity-50');
    if (backdrop) {
      fireEvent.click(backdrop);
      expect(onClose).toHaveBeenCalledTimes(1);
    }
  });

  it('does not render when closed', () => {
    const onClose = vi.fn();
    const { container } = render(
      <BottomSheet isOpen={false} onClose={onClose}>
        <div>Content</div>
      </BottomSheet>
    );

    const sheet = container.querySelector('.translate-y-0');
    expect(sheet).not.toBeInTheDocument();
  });

  it('handles touch events for swipe gesture', () => {
    const onClose = vi.fn();
    const { container } = render(
      <BottomSheet isOpen={true} onClose={onClose}>
        <div>Content</div>
      </BottomSheet>
    );

    const dragHandle = container.querySelector('.cursor-grab');
    expect(dragHandle).toBeInTheDocument();

    if (dragHandle) {
      // Simulate touch start
      fireEvent.touchStart(dragHandle, {
        touches: [{ clientY: 100 }],
      });

      // Simulate touch move (drag down by 150px - should trigger close)
      fireEvent.touchMove(dragHandle, {
        touches: [{ clientY: 250 }],
      });

      // Simulate touch end
      fireEvent.touchEnd(dragHandle);

      expect(onClose).toHaveBeenCalled();
    }
  });

  it('accepts custom maxHeight prop', () => {
    const onClose = vi.fn();
    const { container } = render(
      <BottomSheet isOpen={true} onClose={onClose} maxHeight="60vh">
        <div>Content</div>
      </BottomSheet>
    );

    const sheet = container.querySelector('.rounded-t-2xl');
    expect(sheet).toHaveStyle({ maxHeight: '60vh' });
  });
});

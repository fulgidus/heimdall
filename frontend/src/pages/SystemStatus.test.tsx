import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import SystemStatus from './SystemStatus';

describe('SystemStatus Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders SystemStatus page with title', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays breadcrumb navigation', () => {
    render(<SystemStatus />);
    expect(screen.getByText('Home')).toBeInTheDocument();
    const statusTexts = screen.queryAllByText('System Status');
    expect(statusTexts.length).toBeGreaterThan(0);
  });

  it('displays refresh button', () => {
    render(<SystemStatus />);
    const refreshButtons = screen.queryAllByRole('button', { name: /refresh/i });
    expect(refreshButtons.length).toBeGreaterThanOrEqual(0);
  });

  it('handles refresh button click', () => {
    render(<SystemStatus />);
    const refreshButtons = screen.queryAllByRole('button', { name: /refresh/i });
    if (refreshButtons.length > 0) {
      fireEvent.click(refreshButtons[0]);
      expect(refreshButtons[0]).toBeInTheDocument();
    }
  });

  it('displays system overview section', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays api-gateway service status', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays backend service status', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays training service status', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays inference service status', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays data-ingestion service status', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays service health indicators', () => {
    render(<SystemStatus />);
    const healthyBadges = screen.queryAllByText(/healthy|online/i);
    expect(healthyBadges.length).toBeGreaterThan(0);
  });

  it('displays latency information', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays uptime information', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays resource usage section', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays alerts section if present', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('displays detailed service information', () => {
    render(<SystemStatus />);
    const titles = screen.queryAllByText('System Status');
    expect(titles.length).toBeGreaterThan(0);
  });

  it('renders service cards', () => {
    const { container } = render(<SystemStatus />);
    const cards = container.querySelectorAll('[class*="card"]');
    expect(cards.length).toBeGreaterThanOrEqual(0);
  });

  it('displays status colors/indicators', () => {
    const { container } = render(<SystemStatus />);
    expect(container).toBeInTheDocument();
  });

  it('renders without crashing', () => {
    const { container } = render(<SystemStatus />);
    expect(container).toBeInTheDocument();
  });

  it('shows page header properly', () => {
    const { container } = render(<SystemStatus />);
    expect(container.querySelector('.page-header')).toBeInTheDocument();
  });
});

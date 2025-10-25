import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import StatCard from '../StatCard';

describe('StatCard Component', () => {
    it('renders without crashing', () => {
        render(<StatCard title="Test Title" value="42" />);
        expect(screen.getByText('Test Title')).toBeInTheDocument();
        expect(screen.getByText('42')).toBeInTheDocument();
    });

    it('displays title and value correctly', () => {
        render(<StatCard title="Active WebSDRs" value={7} />);
        expect(screen.getByText('Active WebSDRs')).toBeInTheDocument();
        expect(screen.getByText('7')).toBeInTheDocument();
    });

    it('renders with primary variant by default', () => {
        const { container } = render(<StatCard title="Test" value="100" />);
        const card = container.querySelector('.stat-card-primary');
        expect(card).toBeInTheDocument();
    });

    it('applies success variant class correctly', () => {
        const { container } = render(<StatCard title="Test" value="100" variant="success" />);
        const card = container.querySelector('.stat-card-success');
        expect(card).toBeInTheDocument();
    });

    it('applies warning variant class correctly', () => {
        const { container } = render(<StatCard title="Test" value="100" variant="warning" />);
        const card = container.querySelector('.stat-card-warning');
        expect(card).toBeInTheDocument();
    });

    it('applies danger variant class correctly', () => {
        const { container } = render(<StatCard title="Test" value="100" variant="danger" />);
        const card = container.querySelector('.stat-card-danger');
        expect(card).toBeInTheDocument();
    });

    it('applies info variant class correctly', () => {
        const { container } = render(<StatCard title="Test" value="100" variant="info" />);
        const card = container.querySelector('.stat-card-info');
        expect(card).toBeInTheDocument();
    });

    it('displays subtitle when provided', () => {
        render(<StatCard title="Test" value="100" subtitle="Additional info" />);
        expect(screen.getByText('Additional info')).toBeInTheDocument();
    });

    it('does not display subtitle when not provided', () => {
        render(<StatCard title="Test" value="100" />);
        const subtitle = screen.queryByText(/Additional info/);
        expect(subtitle).not.toBeInTheDocument();
    });

    it('renders icon when provided', () => {
        const { container } = render(<StatCard title="Test" value="100" icon="ph-activity" />);
        const icon = container.querySelector('.ph-activity');
        expect(icon).toBeInTheDocument();
    });

    it('does not render icon when not provided', () => {
        const { container } = render(<StatCard title="Test" value="100" />);
        const icon = container.querySelector('.stat-card-icon');
        expect(icon).not.toBeInTheDocument();
    });

    it('displays upward trend correctly', () => {
        const trend = { value: 12.5, direction: 'up' as const, label: 'vs last week' };
        render(<StatCard title="Test" value="100" trend={trend} />);
        expect(screen.getByText('12.5%')).toBeInTheDocument();
        expect(screen.getByText('vs last week')).toBeInTheDocument();
    });

    it('displays downward trend correctly', () => {
        const trend = { value: -8.3, direction: 'down' as const, label: 'vs last month' };
        render(<StatCard title="Test" value="100" trend={trend} />);
        expect(screen.getByText('8.3%')).toBeInTheDocument();
        expect(screen.getByText('vs last month')).toBeInTheDocument();
    });

    it('displays trend without label', () => {
        const trend = { value: 5.5, direction: 'up' as const };
        render(<StatCard title="Test" value="100" trend={trend} />);
        expect(screen.getByText('5.5%')).toBeInTheDocument();
    });

    it('applies custom className when provided', () => {
        const { container } = render(<StatCard title="Test" value="100" className="custom-class" />);
        const card = container.querySelector('.custom-class');
        expect(card).toBeInTheDocument();
    });

    it('formats large numbers correctly', () => {
        render(<StatCard title="Total Signals" value={1234567} />);
        expect(screen.getByText('1234567')).toBeInTheDocument();
    });

    it('handles string values correctly', () => {
        render(<StatCard title="Status" value="Online" />);
        expect(screen.getByText('Online')).toBeInTheDocument();
    });

    it('handles decimal values correctly', () => {
        render(<StatCard title="Accuracy" value="95.2%" />);
        expect(screen.getByText('95.2%')).toBeInTheDocument();
    });
});

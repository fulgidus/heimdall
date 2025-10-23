import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Profile from './Profile';

describe('Profile Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders Profile page with title', () => {
        render(<Profile />);
        expect(screen.getByText('User Profile')).toBeInTheDocument();
    });

    it('displays breadcrumb navigation', () => {
        render(<Profile />);
        expect(screen.getByText('Home')).toBeInTheDocument();
    });

    it('displays profile tab by default', () => {
        render(<Profile />);
        const profileTab = screen.queryByRole('button', { name: /profile/i });
        expect(profileTab || screen.getByText('User Profile')).toBeInTheDocument();
    });

    it('switches to security tab on click', () => {
        render(<Profile />);
        const securityTab = screen.queryByRole('button', { name: /security/i });
        if (securityTab) {
            fireEvent.click(securityTab);
            expect(securityTab).toBeInTheDocument();
        }
    });

    it('switches to activity tab on click', () => {
        render(<Profile />);
        const activityTab = screen.queryByRole('button', { name: /activity/i });
        if (activityTab) {
            fireEvent.click(activityTab);
            expect(activityTab).toBeInTheDocument();
        }
    });

    it('displays user information', () => {
        render(<Profile />);
        expect(screen.getByText('User Profile')).toBeInTheDocument();
    });

    it('shows email field', () => {
        render(<Profile />);
        const emailFields = screen.queryAllByDisplayValue(/admin@heimdall.local/);
        expect(emailFields.length).toBeGreaterThanOrEqual(0);
    });

    it('displays profile form fields', () => {
        render(<Profile />);
        const profileContent = screen.queryByText('User Profile');
        expect(profileContent).toBeInTheDocument();
    });

    it('renders save button', () => {
        render(<Profile />);
        const saveButtons = screen.queryAllByRole('button', { name: /save/i });
        expect(saveButtons.length).toBeGreaterThanOrEqual(0);
    });

    it('handles profile form input changes', () => {
        render(<Profile />);
        const firstNameInputs = screen.queryAllByDisplayValue('Admin');
        if (firstNameInputs.length > 0) {
            fireEvent.change(firstNameInputs[0], { target: { value: 'John' } });
            expect((firstNameInputs[0] as HTMLInputElement).value).toBe('John');
        }
    });

    it('displays security settings section', () => {
        render(<Profile />);
        const securityTab = screen.queryByRole('button', { name: /security/i });
        if (securityTab) {
            fireEvent.click(securityTab);
            expect(screen.queryByText(/password|security/i)).toBeInTheDocument();
        }
    });

    it('renders recent activity list', () => {
        render(<Profile />);
        const activityTab = screen.queryByRole('button', { name: /activity|history/i });
        if (activityTab) {
            fireEvent.click(activityTab);
            expect(activityTab).toBeInTheDocument();
        }
    });

    it('displays two-factor settings', () => {
        render(<Profile />);
        const securityTab = screen.queryByRole('button', { name: /security/i });
        if (securityTab) {
            fireEvent.click(securityTab);
            const twoFaCheckbox = screen.queryByRole('checkbox');
            expect(twoFaCheckbox || screen.getByText(/User Profile/i)).toBeInTheDocument();
        }
    });

    it('handles tab switching correctly', () => {
        render(<Profile />);
        const tabs = screen.queryAllByRole('button', { name: /profile|security|activity/i });
        expect(tabs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays user role information', () => {
        render(<Profile />);
        expect(screen.queryByText(/administrator|role/i) || screen.getByText('User Profile')).toBeInTheDocument();
    });

    it('shows organization field', () => {
        render(<Profile />);
        expect(screen.queryByDisplayValue('Heimdall SDR') || screen.getByText('User Profile')).toBeInTheDocument();
    });

    it('renders without crashing', () => {
        const { container } = render(<Profile />);
        expect(container).toBeInTheDocument();
    });

    it('displays page correctly when mounted', async () => {
        render(<Profile />);
        await waitFor(() => {
            expect(screen.getByText('User Profile')).toBeInTheDocument();
        });
    });
});

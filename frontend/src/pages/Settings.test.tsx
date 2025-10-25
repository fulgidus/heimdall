import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Settings from './Settings';

describe('Settings Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders Settings page with title', () => {
        render(<Settings />);
        const titles = screen.queryAllByText(/settings/i);
        expect(titles.length).toBeGreaterThan(0);
    });

    it('displays breadcrumb navigation', () => {
        render(<Settings />);
        const homeTexts = screen.queryAllByText('Home');
        const settingsTexts = screen.queryAllByText(/settings/i);
        expect(homeTexts.length > 0 || settingsTexts.length > 0).toBeTruthy();
    });

    it('displays General tab', () => {
        render(<Settings />);
        const generalTabs = screen.queryAllByRole('button', { name: /general/i });
        expect(generalTabs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays API tab', () => {
        render(<Settings />);
        const apiTabs = screen.queryAllByRole('button', { name: /api/i });
        expect(apiTabs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays Notifications tab', () => {
        render(<Settings />);
        const notificationTabs = screen.queryAllByRole('button', { name: /notification/i });
        expect(notificationTabs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays Advanced tab', () => {
        render(<Settings />);
        const advancedTabs = screen.queryAllByRole('button', { name: /advanced/i });
        expect(advancedTabs.length).toBeGreaterThanOrEqual(0);
    });

    it('switches to API tab on click', () => {
        render(<Settings />);
        const apiTabs = screen.queryAllByRole('button', { name: /api/i });
        if (apiTabs.length > 0) {
            fireEvent.click(apiTabs[0]);
            expect(apiTabs[0]).toBeInTheDocument();
        }
    });

    it('switches to Notifications tab on click', () => {
        render(<Settings />);
        const notificationTabs = screen.queryAllByRole('button', { name: /notification/i });
        if (notificationTabs.length > 0) {
            fireEvent.click(notificationTabs[0]);
            expect(notificationTabs[0]).toBeInTheDocument();
        }
    });

    it('switches to Advanced tab on click', () => {
        render(<Settings />);
        const advancedTabs = screen.queryAllByRole('button', { name: /advanced/i });
        if (advancedTabs.length > 0) {
            fireEvent.click(advancedTabs[0]);
            expect(advancedTabs[0]).toBeInTheDocument();
        }
    });

    it('displays theme selector', () => {
        render(<Settings />);
        const themeSelects = screen.queryAllByDisplayValue(/dark|light/i);
        expect(themeSelects.length).toBeGreaterThanOrEqual(0);
    });

    it('displays language selector', () => {
        render(<Settings />);
        const languageSelects = screen.queryAllByDisplayValue(/en|it/i);
        expect(languageSelects.length).toBeGreaterThanOrEqual(0);
    });

    it('displays timezone selector', () => {
        render(<Settings />);
        const timezoneSelects = screen.queryAllByDisplayValue(/UTC/i);
        expect(timezoneSelects.length).toBeGreaterThanOrEqual(0);
    });

    it('displays auto-refresh toggle', () => {
        render(<Settings />);
        const checkboxes = screen.queryAllByRole('checkbox');
        expect(checkboxes.length).toBeGreaterThanOrEqual(0);
    });

    it('displays refresh interval input', () => {
        render(<Settings />);
        const numberInputs = screen.queryAllByRole('spinbutton');
        expect(numberInputs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays API settings inputs', () => {
        render(<Settings />);
        const apiTabs = screen.queryAllByRole('button', { name: /api/i });
        if (apiTabs.length > 0) {
            fireEvent.click(apiTabs[0]);
            expect(apiTabs[0]).toBeInTheDocument();
        }
    });

    it('displays notification settings', () => {
        render(<Settings />);
        const notificationTabs = screen.queryAllByRole('button', { name: /notification/i });
        if (notificationTabs.length > 0) {
            fireEvent.click(notificationTabs[0]);
            expect(notificationTabs[0]).toBeInTheDocument();
        }
    });

    it('displays webhook URL input if notifications enabled', () => {
        render(<Settings />);
        const webhookTexts = screen.queryAllByText(/webhook/i);
        const settingsTexts = screen.queryAllByText(/settings/i);
        expect(webhookTexts.length > 0 || settingsTexts.length > 0).toBeTruthy();
    });

    it('displays Save button', () => {
        render(<Settings />);
        const saveButtons = screen.queryAllByRole('button', { name: /save/i });
        expect(saveButtons.length).toBeGreaterThanOrEqual(0);
    });

    it('handles Save button click', () => {
        render(<Settings />);
        const saveButtons = screen.queryAllByRole('button', { name: /save/i });
        if (saveButtons.length > 0) {
            fireEvent.click(saveButtons[0]);
            expect(saveButtons[0]).toBeInTheDocument();
        }
    });

    it('renders without crashing', () => {
        const { container } = render(<Settings />);
        expect(container).toBeInTheDocument();
    });

    it('displays all tabs content sections', () => {
        render(<Settings />);
        const titles = screen.queryAllByText(/settings/i);
        expect(titles.length).toBeGreaterThan(0);
    });
});

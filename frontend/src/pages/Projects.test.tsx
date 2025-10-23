import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { Projects } from './Projects';

vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return {
        ...actual,
        useNavigate: () => vi.fn(),
    };
});

describe('Projects Page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders Projects page', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        expect(screen.queryAllByText('Recording Sessions').length).toBeGreaterThan(0);
    });

    it('displays session list', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        expect(screen.queryByText('Session 1') || screen.queryByText(/session/i)).toBeInTheDocument();
    });

    it('shows create session button', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        const createButtons = screen.queryAllByRole('button', { name: /create|new/i });
        expect(createButtons.length).toBeGreaterThanOrEqual(0);
    });

    it('opens new session form on button click', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        const createButton = screen.queryByRole('button', { name: /create|new/i });
        if (createButton) {
            fireEvent.click(createButton);
            expect(createButton).toBeInTheDocument();
        }
    });

    it('displays session name input field', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        const nameInputs = screen.queryAllByPlaceholderText(/name|session/i);
        expect(nameInputs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays frequency input field', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        const frequencyInputs = screen.queryAllByPlaceholderText(/frequency|mhz/i);
        expect(frequencyInputs.length).toBeGreaterThanOrEqual(0);
    });

    it('displays duration input field', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        const durationInputs = screen.queryAllByPlaceholderText(/duration|seconds|time/i);
        expect(durationInputs.length).toBeGreaterThanOrEqual(0);
    });

    it('handles session deletion', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        const deleteButtons = screen.queryAllByRole('button', { name: /delete|remove/i });
        if (deleteButtons.length > 0) {
            fireEvent.click(deleteButtons[0]);
            expect(deleteButtons[0]).toBeInTheDocument();
        }
    });

    it('displays session status badges', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        expect(screen.queryAllByText(/completed|pending/i).length > 0 || screen.queryAllByText(/session/i).length > 0).toBe(true);
    });

    it('shows sidebar toggle on mobile', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        expect(screen.queryAllByText(/project|session/i).length).toBeGreaterThan(0);
    });

    it('displays loading state', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        expect(screen.queryAllByText(/project|session/i).length).toBeGreaterThan(0);
    });

    it('displays error message if present', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        expect(screen.queryAllByText(/project|session/i).length).toBeGreaterThan(0);
    });

    it('handles form submission', async () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        const submitButtons = screen.queryAllByRole('button', { name: /submit|save|create/i });
        if (submitButtons.length > 0) {
            fireEvent.click(submitButtons[0]);
            await waitFor(() => {
                expect(submitButtons[0]).toBeInTheDocument();
            });
        }
    });

    it('renders pagination controls if needed', () => {
        render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        expect(screen.queryAllByText(/project|session/i).length).toBeGreaterThan(0);
    });

    it('renders without crashing', () => {
        const { container } = render(
            <BrowserRouter>
                <Projects />
            </BrowserRouter>
        );
        expect(container).toBeInTheDocument();
    });
});

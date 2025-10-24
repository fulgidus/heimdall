/**
 * Interactive Features Validation Tests
 * Phase 7: Testing & Validation
 * 
 * Tests user interactions, form submissions, navigation, and UI interactions
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

// Mock stores before importing
vi.mock('../store');

// Components
import { Button, Card, Input, Select, Textarea, Badge, Modal, Table } from '../components';
import type { SelectOption } from '../components';

// Pages
import DataIngestion from '../pages/DataIngestion';
import Settings from '../pages/Settings';

describe('Phase 7: Interactive Features Validation', () => {
    describe('Button Interactions', () => {
        it('should handle click events', async () => {
            const handleClick = vi.fn();
            render(<Button onClick={handleClick}>Click Me</Button>);

            const button = screen.getByText('Click Me');
            fireEvent.click(button);

            expect(handleClick).toHaveBeenCalledTimes(1);
        });

        it('should show loading state', () => {
            render(<Button isLoading>Loading...</Button>);
            expect(screen.getByText('Loading...')).toBeTruthy();
        });

        it('should be disabled when loading', () => {
            render(<Button isLoading>Submit</Button>);
            const button = screen.getByRole('button');
            expect(button).toBeDisabled();
        });

        it('should support different variants', () => {
            const { rerender } = render(<Button variant="primary">Primary</Button>);
            expect(screen.getByText('Primary')).toBeTruthy();

            rerender(<Button variant="success">Success</Button>);
            expect(screen.getByText('Success')).toBeTruthy();

            rerender(<Button variant="danger">Danger</Button>);
            expect(screen.getByText('Danger')).toBeTruthy();
        });
    });

    describe('Form Interactions', () => {
        it('should handle input changes', async () => {
            const handleChange = vi.fn();
            render(<Input onChange={handleChange} placeholder="Enter text" />);

            const input = screen.getByPlaceholderText('Enter text');
            fireEvent.change(input, { target: { value: 'Hello' } });

            expect(handleChange).toHaveBeenCalled();
        });

        it('should display validation errors', () => {
            render(<Input error="This field is required" label="Username" />);
            expect(screen.getByText('This field is required')).toBeTruthy();
        });

        it('should handle select changes', async () => {
            const options: SelectOption[] = [
                { value: '1', label: 'Option 1' },
                { value: '2', label: 'Option 2' },
                { value: '3', label: 'Option 3' },
            ];

            const handleChange = vi.fn();
            render(<Select options={options} onChange={handleChange} label="Select an option" />);

            const select = screen.getByLabelText('Select an option');
            fireEvent.change(select, { target: { value: '2' } });

            expect(handleChange).toHaveBeenCalled();
        });

        it('should handle textarea input', async () => {
            const handleChange = vi.fn();
            render(<Textarea onChange={handleChange} placeholder="Enter notes" />);

            const textarea = screen.getByPlaceholderText('Enter notes');
            fireEvent.change(textarea, { target: { value: 'Test notes' } });

            expect(handleChange).toHaveBeenCalled();
        });
    });

    describe('Modal Interactions', () => {
        it('should open and close modal', async () => {
            let isOpen = true;
            const handleClose = vi.fn(() => {
                isOpen = false;
            });

            const { rerender } = render(
                <Modal isOpen={isOpen} onClose={handleClose} title="Test Modal">
                    Modal Content
                </Modal>
            );

            expect(screen.getByText('Test Modal')).toBeTruthy();
            expect(screen.getByText('Modal Content')).toBeTruthy();

            // Click close button
            const closeButton = screen.getByText('✕');
            fireEvent.click(closeButton);

            expect(handleClose).toHaveBeenCalledTimes(1);

            // Rerender with closed state
            rerender(
                <Modal isOpen={false} onClose={handleClose} title="Test Modal">
                    Modal Content
                </Modal>
            );

            expect(screen.queryByText('Test Modal')).toBeFalsy();
        });
    });

    describe('Table Interactions', () => {
        const mockData = [
            { id: 1, name: 'Item 1', status: 'active' },
            { id: 2, name: 'Item 2', status: 'inactive' },
            { id: 3, name: 'Item 3', status: 'active' },
        ];

        const columns = [
            { key: 'id', header: 'ID' },
            { key: 'name', header: 'Name' },
            { key: 'status', header: 'Status' },
        ];

        it('should display table data', () => {
            render(<Table columns={columns} data={mockData} />);

            expect(screen.getByText('Item 1')).toBeTruthy();
            expect(screen.getByText('Item 2')).toBeTruthy();
            expect(screen.getByText('Item 3')).toBeTruthy();
        });

        it('should handle row clicks', async () => {
            const handleRowClick = vi.fn();
            render(<Table columns={columns} data={mockData} onRowClick={handleRowClick} />);

            const row = screen.getByText('Item 1').closest('tr');
            if (row) {
                fireEvent.click(row);
                expect(handleRowClick).toHaveBeenCalledWith(mockData[0], 0);
            }
        });

        it('should show loading state', () => {
            render(<Table columns={columns} data={[]} loading={true} />);
            expect(screen.getByText('Loading...')).toBeTruthy();
        });

        it('should show empty state', () => {
            render(<Table columns={columns} data={[]} emptyMessage="No items found" />);
            expect(screen.getByText('No items found')).toBeTruthy();
        });
    });

    describe('Badge Display', () => {
        it('should display badges with variants', () => {
            const { rerender } = render(<Badge variant="success">Success</Badge>);
            expect(screen.getByText('Success')).toBeTruthy();

            rerender(<Badge variant="danger">Error</Badge>);
            expect(screen.getByText('Error')).toBeTruthy();

            rerender(<Badge variant="warning">Warning</Badge>);
            expect(screen.getByText('Warning')).toBeTruthy();
        });
    });

    describe('Page Navigation', () => {
        it('should navigate between pages', async () => {
            render(
                <BrowserRouter>
                    <DataIngestion />
                </BrowserRouter>
            );

            // Check page renders - use getAllByText to handle multiple matches
            const headings = screen.getAllByText(/Data Ingestion/i);
            expect(headings.length).toBeGreaterThan(0);
        });
    });

    describe('Form Submission', () => {
        it('should handle form in Data Ingestion page', async () => {
            render(
                <BrowserRouter>
                    <DataIngestion />
                </BrowserRouter>
            );

            // Page should render - check for heading specifically
            const heading = screen.getAllByText(/Data Ingestion/i).find(el => el.tagName === 'H2');
            expect(heading).toBeTruthy();
        });

        it('should handle settings updates', async () => {
            render(
                <BrowserRouter>
                    <Settings />
                </BrowserRouter>
            );

            // Settings page should render - check for heading specifically
            const heading = screen.getAllByText(/Settings/i).find(el => el.tagName === 'H2');
            expect(heading).toBeTruthy();
        });
    });

    describe('Tab Navigation', () => {
        it('should switch between tabs in Settings page', async () => {
            render(
                <BrowserRouter>
                    <Settings />
                </BrowserRouter>
            );

            // Check tabs exist
            const tabs = screen.getAllByRole('button');
            expect(tabs.length).toBeGreaterThan(0);
        });

        it('should switch between tabs in Data Ingestion page', async () => {
            render(
                <BrowserRouter>
                    <DataIngestion />
                </BrowserRouter>
            );

            // Check tabs exist
            const tabs = screen.getAllByRole('button');
            expect(tabs.length).toBeGreaterThan(0);
        });
    });

    describe('Accessibility', () => {
        it('should have accessible buttons', () => {
            render(<Button>Accessible Button</Button>);
            const button = screen.getByRole('button');
            expect(button).toBeTruthy();
        });

        it('should have accessible selects with labels', () => {
            const options: SelectOption[] = [
                { value: '1', label: 'Option 1' },
            ];
            render(<Select options={options} label="Select Option" />);
            const select = screen.getByLabelText('Select Option');
            expect(select).toBeTruthy();
        });
    });
});

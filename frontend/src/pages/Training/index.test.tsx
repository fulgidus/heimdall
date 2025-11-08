/**
 * Training Components Tests
 *
 * Tests for Training page components
 * Verifies rendering, user interactions, and state management
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Training from './index';
import { useTrainingStore } from '@/store/trainingStore';

// Mock the training store
vi.mock('@/store/trainingStore');

const mockTrainingStore = {
  jobs: [],
  currentJob: null,
  models: [],
  metrics: new Map(),
  isLoading: false,
  error: null,
  wsConnected: false,
  fetchJobs: vi.fn(),
  fetchModels: vi.fn(),
  createJob: vi.fn(),
  pauseJob: vi.fn(),
  resumeJob: vi.fn(),
  cancelJob: vi.fn(),
  deleteJob: vi.fn(),
  deleteModel: vi.fn(),
  downloadModel: vi.fn(),
  importModel: vi.fn(),
  clearError: vi.fn(),
};

describe('Training Page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useTrainingStore).mockReturnValue(mockTrainingStore as any);
  });

  const renderWithRouter = (component: React.ReactElement) => {
    return render(<BrowserRouter>{component}</BrowserRouter>);
  };

  describe('Page Rendering', () => {
    it('should render training page with header', () => {
      renderWithRouter(<Training />);

      expect(screen.getByText('Training')).toBeInTheDocument();
      expect(screen.getByText(/manage and monitor ML training jobs/i)).toBeInTheDocument();
    });

    it('should render all three tabs', () => {
      renderWithRouter(<Training />);

      expect(screen.getByRole('tab', { name: /training jobs/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /metrics/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /trained models/i })).toBeInTheDocument();
    });

    it('should have Training Jobs tab active by default', () => {
      renderWithRouter(<Training />);

      const jobsTab = screen.getByRole('tab', { name: /training jobs/i });
      expect(jobsTab).toHaveAttribute('aria-selected', 'true');
    });
  });

  describe('Tab Navigation', () => {
    it('should switch to Metrics tab when clicked', async () => {
      renderWithRouter(<Training />);

      const metricsTab = screen.getByRole('tab', { name: /metrics/i });
      fireEvent.click(metricsTab);

      await waitFor(() => {
        expect(metricsTab).toHaveAttribute('aria-selected', 'true');
      });
    });

    it('should switch to Trained Models tab when clicked', async () => {
      renderWithRouter(<Training />);

      const modelsTab = screen.getByRole('tab', { name: /trained models/i });
      fireEvent.click(modelsTab);

      await waitFor(() => {
        expect(modelsTab).toHaveAttribute('aria-selected', 'true');
      });
    });
  });

  describe('Data Fetching', () => {
    it('should fetch jobs on mount', () => {
      renderWithRouter(<Training />);

      expect(mockTrainingStore.fetchJobs).toHaveBeenCalledOnce();
    });

    it('should fetch models when switching to Models tab', async () => {
      renderWithRouter(<Training />);

      const modelsTab = screen.getByRole('tab', { name: /trained models/i });
      fireEvent.click(modelsTab);

      await waitFor(() => {
        expect(mockTrainingStore.fetchModels).toHaveBeenCalled();
      });
    });
  });

  describe('Loading State', () => {
    it('should show loading spinner when isLoading is true', () => {
      vi.mocked(useTrainingStore).mockReturnValue({
        ...mockTrainingStore,
        isLoading: true,
      } as any);

      renderWithRouter(<Training />);

      expect(screen.getByRole('status')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should display error message when error exists', () => {
      const errorMessage = 'Failed to fetch training jobs';
      vi.mocked(useTrainingStore).mockReturnValue({
        ...mockTrainingStore,
        error: errorMessage,
      } as any);

      renderWithRouter(<Training />);

      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });

    it('should clear error when dismiss button is clicked', async () => {
      vi.mocked(useTrainingStore).mockReturnValue({
        ...mockTrainingStore,
        error: 'Some error',
      } as any);

      renderWithRouter(<Training />);

      const dismissButton = screen.getByRole('button', { name: /dismiss/i });
      fireEvent.click(dismissButton);

      expect(mockTrainingStore.clearError).toHaveBeenCalled();
    });
  });

  describe('Jobs Tab', () => {
    it('should render empty state when no jobs exist', () => {
      renderWithRouter(<Training />);

      expect(screen.getByText(/no training jobs yet/i)).toBeInTheDocument();
    });

    it('should render job cards when jobs exist', () => {
      const mockJobs = [
        {
          id: 'job-1',
          name: 'Test Job 1',
          status: 'running',
          config: { model_type: 'localization', epochs: 100 },
          created_at: '2025-01-01T00:00:00Z',
          current_epoch: 10,
          total_epochs: 100,
          progress: 0.1,
        },
        {
          id: 'job-2',
          name: 'Test Job 2',
          status: 'completed',
          config: { model_type: 'localization', epochs: 50 },
          created_at: '2025-01-02T00:00:00Z',
          current_epoch: 50,
          total_epochs: 50,
          progress: 1.0,
        },
      ];

      vi.mocked(useTrainingStore).mockReturnValue({
        ...mockTrainingStore,
        jobs: mockJobs,
      } as any);

      renderWithRouter(<Training />);

      expect(screen.getByText('Test Job 1')).toBeInTheDocument();
      expect(screen.getByText('Test Job 2')).toBeInTheDocument();
    });

    it('should open create job dialog when Create New Job button is clicked', async () => {
      renderWithRouter(<Training />);

      const createButton = screen.getByRole('button', { name: /create new job/i });
      fireEvent.click(createButton);

      await waitFor(() => {
        expect(screen.getByRole('dialog')).toBeInTheDocument();
      });
    });
  });

  describe('Models Tab', () => {
    it('should render empty state when no models exist', async () => {
      renderWithRouter(<Training />);

      const modelsTab = screen.getByRole('tab', { name: /trained models/i });
      fireEvent.click(modelsTab);

      await waitFor(() => {
        expect(screen.getByText(/no trained models yet/i)).toBeInTheDocument();
      });
    });

    it('should render model cards when models exist', async () => {
      const mockModels = [
        {
          id: 'model-1',
          name: 'Model v1.0',
          version: '1.0.0',
          job_id: 'job-1',
          created_at: '2025-01-01T00:00:00Z',
          metrics: { accuracy: 0.95, loss: 0.2 },
          size_bytes: 1024000,
        },
      ];

      vi.mocked(useTrainingStore).mockReturnValue({
        ...mockTrainingStore,
        models: mockModels,
      } as any);

      renderWithRouter(<Training />);

      const modelsTab = screen.getByRole('tab', { name: /trained models/i });
      fireEvent.click(modelsTab);

      await waitFor(() => {
        expect(screen.getByText('Model v1.0')).toBeInTheDocument();
      });
    });

    it('should open import dialog when Import Model button is clicked', async () => {
      renderWithRouter(<Training />);

      const modelsTab = screen.getByRole('tab', { name: /trained models/i });
      fireEvent.click(modelsTab);

      await waitFor(() => {
        const importButton = screen.getByRole('button', { name: /import model/i });
        fireEvent.click(importButton);
      });

      await waitFor(() => {
        expect(screen.getByRole('dialog')).toBeInTheDocument();
      });
    });
  });
});

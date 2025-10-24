import { create } from 'zustand';
import type { LocalizationResult } from '@/services/api/types';
import { inferenceService } from '@/services/api';
import type { PredictionRequest, PredictionResponse } from '@/services/api/inference';

interface LocalizationState {
    // Data
    recentLocalizations: LocalizationResult[];
    currentPrediction: PredictionResponse | null;

    // UI State
    isLoading: boolean;
    isPredicting: boolean;
    error: string | null;
    selectedResultId: string | null;

    // Actions
    setLoading: (loading: boolean) => void;
    setPredicting: (predicting: boolean) => void;
    setError: (error: string | null) => void;
    setSelectedResult: (id: string | null) => void;

    // API Actions
    fetchRecentLocalizations: (limit?: number) => Promise<void>;
    predictLocalization: (request: PredictionRequest) => Promise<PredictionResponse>;
    clearCurrentPrediction: () => void;
    refreshData: () => Promise<void>;
}

export const useLocalizationStore = create<LocalizationState>((set, get) => ({
    // Initial state
    recentLocalizations: [],
    currentPrediction: null,
    isLoading: false,
    isPredicting: false,
    error: null,
    selectedResultId: null,

    // Basic setters
    setLoading: (loading) => set({ isLoading: loading }),
    setPredicting: (predicting) => set({ isPredicting: predicting }),
    setError: (error) => set({ error }),
    setSelectedResult: (id) => set({ selectedResultId: id }),

    // API Actions
    fetchRecentLocalizations: async (limit = 10) => {
        set({ isLoading: true, error: null });

        try {
            const localizations = await inferenceService.getRecentLocalizations(limit);

            set({
                recentLocalizations: localizations,
                error: null,
            });
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to fetch localizations';
            set({ error: errorMessage });
            console.error('Failed to fetch recent localizations:', error);
        } finally {
            set({ isLoading: false });
        }
    },

    predictLocalization: async (request: PredictionRequest) => {
        set({ isPredicting: true, error: null });

        try {
            const prediction = await inferenceService.predictLocalization(request);

            set({
                currentPrediction: prediction,
                error: null,
            });

            // Refresh recent localizations to include the new one
            await get().fetchRecentLocalizations();

            return prediction;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Prediction failed';
            set({ error: errorMessage });
            console.error('Prediction failed:', error);
            throw error;
        } finally {
            set({ isPredicting: false });
        }
    },

    clearCurrentPrediction: () => {
        set({ currentPrediction: null });
    },

    refreshData: async () => {
        await get().fetchRecentLocalizations();
    },
}));
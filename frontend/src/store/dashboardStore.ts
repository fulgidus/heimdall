import { create } from 'zustand';

interface DashboardMetrics {
    activeUsers: number;
    totalRevenue: number;
    conversionRate: number;
    performance: number;
}

interface DashboardStore {
    metrics: DashboardMetrics;
    isLoading: boolean;
    setMetrics: (metrics: DashboardMetrics) => void;
    setLoading: (loading: boolean) => void;
    fetchMetrics: () => Promise<void>;
}

export const useDashboardStore = create<DashboardStore>((set) => ({
    metrics: {
        activeUsers: 0,
        totalRevenue: 0,
        conversionRate: 0,
        performance: 0,
    },
    isLoading: false,

    setMetrics: (metrics) => set({ metrics }),
    setLoading: (loading) => set({ isLoading: loading }),

    fetchMetrics: async () => {
        set({ isLoading: true });
        try {
            // Mock API call - replace with actual API
            // const response = await fetch('/api/metrics')
            // const data = await response.json()

            const mockMetrics: DashboardMetrics = {
                activeUsers: 2451,
                totalRevenue: 24567.89,
                conversionRate: 3.24,
                performance: 94.5,
            };

            set({ metrics: mockMetrics });
        } catch (error) {
            console.error('Failed to fetch metrics:', error);
        } finally {
            set({ isLoading: false });
        }
    },
}));

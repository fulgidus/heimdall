import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { WidgetConfig, WidgetType } from '@/types/widgets';
import { AVAILABLE_WIDGETS } from '@/types/widgets';

interface WidgetStore {
    widgets: WidgetConfig[];
    addWidget: (type: WidgetType) => void;
    removeWidget: (id: string) => void;
    updateWidget: (id: string, updates: Partial<WidgetConfig>) => void;
    resetToDefault: () => void;
}

// Default dashboard widgets
const DEFAULT_WIDGETS: WidgetConfig[] = [
    {
        id: 'websdr-status-default',
        type: 'websdr-status',
        title: 'WebSDR Status',
        size: 'large',
        position: 0,
    },
    {
        id: 'system-health-default',
        type: 'system-health',
        title: 'System Health',
        size: 'medium',
        position: 1,
    },
    {
        id: 'model-performance-default',
        type: 'model-performance',
        title: 'Model Performance',
        size: 'medium',
        position: 2,
    },
];

export const useWidgetStore = create<WidgetStore>()(
    persist(
        (set) => ({
            widgets: DEFAULT_WIDGETS,

            addWidget: (type: WidgetType) => {
                const definition = AVAILABLE_WIDGETS.find(w => w.type === type);
                if (!definition) return;

                const newWidget: WidgetConfig = {
                    id: `${type}-${Date.now()}`,
                    type,
                    title: definition.title,
                    size: definition.defaultSize,
                    position: Date.now(),
                };

                set((state) => ({
                    widgets: [...state.widgets, newWidget],
                }));
            },

            removeWidget: (id: string) => {
                set((state) => ({
                    widgets: state.widgets.filter(w => w.id !== id),
                }));
            },

            updateWidget: (id: string, updates: Partial<WidgetConfig>) => {
                set((state) => ({
                    widgets: state.widgets.map(w =>
                        w.id === id ? { ...w, ...updates } : w
                    ),
                }));
            },

            resetToDefault: () => {
                set({ widgets: DEFAULT_WIDGETS });
            },
        }),
        {
            name: 'heimdall-dashboard-widgets',
        }
    )
);

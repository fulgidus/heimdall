/**
 * Widget Types and Interfaces
 *
 * Defines the structure for dashboard widgets
 */

export type WidgetType =
  | 'websdr-status'
  | 'system-health'
  | 'recent-activity'
  | 'signal-chart'
  | 'model-performance'
  | 'quick-actions';

export interface WidgetConfig {
  id: string;
  type: WidgetType;
  title: string;
  size: 'small' | 'medium' | 'large';
  position: number;
  width?: number; // Custom width in Bootstrap columns (1-12), overrides size
  height?: 'auto' | 'small' | 'medium' | 'large' | 'xlarge'; // Custom height
}

export interface WidgetDefinition {
  type: WidgetType;
  title: string;
  description: string;
  defaultSize: 'small' | 'medium' | 'large';
  icon: string;
}

// Widget catalog - available widgets
export const AVAILABLE_WIDGETS: WidgetDefinition[] = [
  {
    type: 'websdr-status',
    title: 'WebSDR Status',
    description: 'Real-time status of all WebSDR receivers',
    defaultSize: 'large',
    icon: 'ph-radio-button',
  },
  {
    type: 'system-health',
    title: 'System Health',
    description: 'Health status of all microservices',
    defaultSize: 'medium',
    icon: 'ph-cpu',
  },
  {
    type: 'recent-activity',
    title: 'Recent Activity',
    description: 'Latest recording sessions and predictions',
    defaultSize: 'medium',
    icon: 'ph-clock-countdown',
  },
  {
    type: 'signal-chart',
    title: 'Signal Detections',
    description: 'Real-time signal detection chart',
    defaultSize: 'large',
    icon: 'ph-chart-line',
  },
  {
    type: 'model-performance',
    title: 'Model Performance',
    description: 'ML model metrics and accuracy',
    defaultSize: 'medium',
    icon: 'ph-brain',
  },
  {
    type: 'quick-actions',
    title: 'Quick Actions',
    description: 'Common tasks and shortcuts',
    defaultSize: 'small',
    icon: 'ph-lightning',
  },
];

// Size to column span mapping (Bootstrap grid)
export const SIZE_TO_COLUMNS: Record<WidgetConfig['size'], number> = {
  small: 4,
  medium: 6,
  large: 12,
};

// Height class mapping
export const HEIGHT_TO_CLASS: Record<Exclude<WidgetConfig['height'], undefined>, string> = {
  auto: '',
  small: 'widget-height-small',
  medium: 'widget-height-medium',
  large: 'widget-height-large',
  xlarge: 'widget-height-xlarge',
};

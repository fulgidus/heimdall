/**
 * UncertaintyCalibrationChart Component
 * 
 * Displays model's predicted uncertainty vs calibration error
 * Well-calibrated models: predicted uncertainty ≈ actual error
 */

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { TrainingMetric } from '../../types';

interface UncertaintyCalibrationChartProps {
  metrics: TrainingMetric[];
}

export const UncertaintyCalibrationChart: React.FC<UncertaintyCalibrationChartProps> = ({ metrics }) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No uncertainty data available</p>
        </div>
      </div>
    );
  }

  // Filter metrics that have uncertainty fields
  const metricsWithUncertainty = metrics.filter(m => m.mean_predicted_uncertainty_km !== undefined);

  if (metricsWithUncertainty.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No uncertainty data available (training with older model)</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-body">
        <h5 className="card-title mb-1">Uncertainty Calibration</h5>
        <p className="text-muted small mb-3">
          Model's predicted uncertainty vs actual error. Goal: calibration error → 0
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={metricsWithUncertainty}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis 
              dataKey="epoch" 
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
              stroke="#6b7280"
            />
            <YAxis 
              label={{ value: 'Distance (km)', angle: -90, position: 'insideLeft' }}
              stroke="#6b7280"
              domain={[0, 'auto']}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #e5e7eb',
                borderRadius: '0.375rem',
              }}
              formatter={(value: number) => `${value.toFixed(3)} km`}
            />
            <Legend />
            
            <Line 
              type="monotone" 
              dataKey="mean_predicted_uncertainty_km" 
              stroke="#8b5cf6" 
              strokeWidth={2}
              name="Predicted Uncertainty (σ)"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line 
              type="monotone" 
              dataKey="uncertainty_calibration_error" 
              stroke="#ef4444" 
              strokeWidth={2}
              name="Calibration Error"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line 
              type="monotone" 
              dataKey="val_rmse_km" 
              stroke="#10b981" 
              strokeWidth={1.5}
              strokeDasharray="5 5"
              name="Actual Error (RMSE)"
              dot={{ r: 2 }}
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="mt-2">
          <small className="text-muted">
            💡 Well-calibrated models have predicted uncertainty ≈ actual error (calibration error near 0)
          </small>
        </div>
      </div>
    </div>
  );
};

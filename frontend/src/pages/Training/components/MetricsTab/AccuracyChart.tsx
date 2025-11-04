/**
 * AccuracyChart Component
 * 
 * Displays training and validation RMSE (Root Mean Square Error) in meters
 * All metrics follow SI units (meters)
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

interface AccuracyChartProps {
  metrics: TrainingMetric[];
}

export const AccuracyChart: React.FC<AccuracyChartProps> = ({ metrics }) => {
  // Filter metrics that have RMSE data (in meters - SI unit)
  const metricsWithRmse = metrics.filter(m => 
    m.train_rmse_m !== undefined && m.val_rmse_m !== undefined
  );

  if (!metricsWithRmse || metricsWithRmse.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No RMSE data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-body">
        <h5 className="card-title mb-3">Training & Validation RMSE (meters)</h5>
        <ResponsiveContainer width="100%" height={300}>
        <LineChart data={metricsWithRmse}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="epoch" 
            label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
            stroke="#6b7280"
          />
          <YAxis 
            label={{ value: 'RMSE (m)', angle: -90, position: 'insideLeft' }}
            stroke="#6b7280"
            scale="log"
            domain={['auto', 'auto']}
            allowDataOverflow={false}
            tickFormatter={(value) => `${value.toFixed(0)}m`}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#fff', 
              border: '1px solid #e5e7eb',
              borderRadius: '0.375rem',
            }}
            formatter={(value: number) => `${value.toFixed(2)}m`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="train_rmse_m" 
            stroke="#10b981" 
            strokeWidth={2}
            name="Train RMSE"
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line 
            type="monotone" 
            dataKey="val_rmse_m" 
            stroke="#f59e0b" 
            strokeWidth={2}
            name="Validation RMSE"
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
};

/**
 * GradientHealthChart Component
 * 
 * Displays training health metrics: gradient norm and weight norm over epochs
 * Helps detect gradient explosion/vanishing and monitor model parameter growth
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

interface GradientHealthChartProps {
  metrics: TrainingMetric[];
}

export const GradientHealthChart: React.FC<GradientHealthChartProps> = ({ metrics }) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No gradient data available</p>
        </div>
      </div>
    );
  }

  // Filter metrics that have gradient/weight norm fields
  const metricsWithNorms = metrics.filter(m => m.gradient_norm !== undefined);

  if (metricsWithNorms.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No gradient data available (training with older model)</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-body">
        <h5 className="card-title mb-1">Training Health (Norms)</h5>
        <p className="text-muted small mb-3">
          Monitor gradient and weight norms. Sudden spikes indicate training instability
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={metricsWithNorms}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis 
              dataKey="epoch" 
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
              stroke="#6b7280"
            />
            <YAxis 
              yAxisId="left"
              label={{ value: 'Gradient Norm (L2)', angle: -90, position: 'insideLeft' }}
              stroke="#3b82f6"
              domain={[0, 'auto']}
            />
            <YAxis 
              yAxisId="right"
              orientation="right"
              label={{ value: 'Weight Norm (L2)', angle: 90, position: 'insideRight' }}
              stroke="#10b981"
              domain={[0, 'auto']}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #e5e7eb',
                borderRadius: '0.375rem',
              }}
              formatter={(value: number) => value.toFixed(4)}
            />
            <Legend />
            
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="gradient_norm" 
              stroke="#3b82f6" 
              strokeWidth={2}
              name="Gradient Norm"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="weight_norm" 
              stroke="#10b981" 
              strokeWidth={2}
              name="Weight Norm"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="mt-2">
          <small className="text-muted">
            💡 Stable training: gradients stay bounded, weights grow slowly and stabilize
          </small>
        </div>
      </div>
    </div>
  );
};

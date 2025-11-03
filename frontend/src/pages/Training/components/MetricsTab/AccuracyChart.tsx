/**
 * AccuracyChart Component
 * 
 * Displays training and validation accuracy over epochs using Recharts
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
  // Filter metrics that have accuracy data
  const metricsWithAccuracy = metrics.filter(m => 
    m.train_accuracy !== undefined && m.val_accuracy !== undefined
  );

  if (!metricsWithAccuracy || metricsWithAccuracy.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No accuracy data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-body">
        <h5 className="card-title mb-3">Training & Validation Accuracy</h5>
        <ResponsiveContainer width="100%" height={300}>
        <LineChart data={metricsWithAccuracy}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="epoch" 
            label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
            stroke="#6b7280"
          />
          <YAxis 
            label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }}
            stroke="#6b7280"
            domain={[0, 1]}
            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#fff', 
              border: '1px solid #e5e7eb',
              borderRadius: '0.375rem',
            }}
            formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="train_accuracy" 
            stroke="#10b981" 
            strokeWidth={2}
            name="Train Accuracy"
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line 
            type="monotone" 
            dataKey="val_accuracy" 
            stroke="#f59e0b" 
            strokeWidth={2}
            name="Validation Accuracy"
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
};

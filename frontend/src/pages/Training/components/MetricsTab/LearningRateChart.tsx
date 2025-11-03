/**
 * LearningRateChart Component
 * 
 * Displays learning rate over epochs using Recharts
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

interface LearningRateChartProps {
  metrics: TrainingMetric[];
}

export const LearningRateChart: React.FC<LearningRateChartProps> = ({ metrics }) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No learning rate data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-body">
        <h5 className="card-title mb-3">Learning Rate Schedule</h5>
        <ResponsiveContainer width="100%" height={300}>
        <LineChart data={metrics}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="epoch" 
            label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
            stroke="#6b7280"
          />
          <YAxis 
            label={{ value: 'Learning Rate', angle: -90, position: 'insideLeft' }}
            stroke="#6b7280"
            scale="log"
            domain={['auto', 'auto']}
            tickFormatter={(value) => value.toExponential(2)}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#fff', 
              border: '1px solid #e5e7eb',
              borderRadius: '0.375rem',
            }}
            formatter={(value: number) => value.toExponential(6)}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="learning_rate" 
            stroke="#8b5cf6" 
            strokeWidth={2}
            name="Learning Rate"
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
};

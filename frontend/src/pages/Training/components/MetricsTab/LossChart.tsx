/**
 * LossChart Component
 * 
 * Displays training and validation loss over epochs using Recharts
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

interface LossChartProps {
  metrics: TrainingMetric[];
}

export const LossChart: React.FC<LossChartProps> = ({ metrics }) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No loss data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-body">
        <h5 className="card-title mb-3">Training & Validation Loss</h5>
        <ResponsiveContainer width="100%" height={300}>
        <LineChart data={metrics}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="epoch" 
            label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
            stroke="#6b7280"
          />
          <YAxis 
            label={{ value: 'Loss (log scale)', angle: -90, position: 'insideLeft' }}
            stroke="#6b7280"
            scale="log"
            domain={['auto', 'auto']}
            allowDataOverflow={false}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#fff', 
              border: '1px solid #e5e7eb',
              borderRadius: '0.375rem',
            }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="train_loss" 
            stroke="#3b82f6" 
            strokeWidth={2}
            name="Train Loss"
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line 
            type="monotone" 
            dataKey="val_loss" 
            stroke="#ef4444" 
            strokeWidth={2}
            name="Validation Loss"
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
};

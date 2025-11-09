/**
 * DistanceErrorChart Component
 * 
 * Displays RMSE distance errors (train, val, val_good_geom) over epochs
 * Shows localization accuracy in meters (SI unit) - key metric for the project
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
  ReferenceLine,
} from 'recharts';
import type { TrainingMetric } from '../../types';

interface DistanceErrorChartProps {
  metrics: TrainingMetric[];
}

export const DistanceErrorChart: React.FC<DistanceErrorChartProps> = ({ metrics }) => {
  console.log('📈 DistanceErrorChart - metrics received:', { 
    isArray: Array.isArray(metrics), 
    length: metrics?.length,
    firstItem: metrics?.[0],
    hasTrainRmse: metrics?.[0]?.train_rmse_m !== undefined
  });

  if (!metrics || metrics.length === 0) {
    console.log('⚠️ DistanceErrorChart - No metrics array');
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No distance error data available</p>
        </div>
      </div>
    );
  }

  // Filter metrics that have the new distance fields
  const metricsWithDistance = metrics.filter(m => m.train_rmse_m !== undefined);
  console.log('🔍 DistanceErrorChart - filtered metrics:', { 
    original: metrics.length, 
    filtered: metricsWithDistance.length 
  });

  if (metricsWithDistance.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No distance error data available (training with older model)</p>
        </div>
      </div>
    );
  }

  // Project KPI: ±30m @ 68% confidence
  const projectKpiMeters = 30;

  return (
    <div className="card">
      <div className="card-body">
        <h5 className="card-title mb-1">Distance Error (RMSE)</h5>
        <p className="text-muted small mb-3">
          Localization accuracy in meters (SI unit). Goal: &lt;30m for validation
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={metricsWithDistance}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis 
              dataKey="epoch" 
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
              stroke="#6b7280"
            />
            <YAxis 
              label={{ value: 'RMSE (m)', angle: -90, position: 'insideLeft' }}
              stroke="#6b7280"
              domain={[0, 'auto']}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #e5e7eb',
                borderRadius: '0.375rem',
              }}
              formatter={(value: number) => `${value.toFixed(1)} m`}
            />
            <Legend />
            
            {/* Project KPI reference line */}
            <ReferenceLine 
              y={projectKpiMeters} 
              stroke="#10b981" 
              strokeDasharray="5 5" 
              label={{ value: 'Target (30m)', position: 'right', fill: '#10b981' }}
            />
            
            <Line 
              type="monotone" 
              dataKey="train_rmse_m" 
              stroke="#3b82f6" 
              strokeWidth={2}
              name="Train RMSE"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line 
              type="monotone" 
              dataKey="val_rmse_m" 
              stroke="#ef4444" 
              strokeWidth={2}
              name="Val RMSE"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line 
              type="monotone" 
              dataKey="val_rmse_good_geom_m" 
              stroke="#10b981" 
              strokeWidth={2}
              name="Val RMSE (GDOP<5)"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

/**
 * DistancePercentilesChart Component
 * 
 * Displays distance error percentiles (p50, p68, p95) over epochs
 * p68 is the PROJECT KPI: ±30m @ 68% confidence
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

interface DistancePercentilesChartProps {
  metrics: TrainingMetric[];
}

export const DistancePercentilesChart: React.FC<DistancePercentilesChartProps> = ({ metrics }) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No percentile data available</p>
        </div>
      </div>
    );
  }

  // Filter metrics that have the new percentile fields
  const metricsWithPercentiles = metrics.filter(m => m.val_distance_p68_km !== undefined);

  if (metricsWithPercentiles.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No percentile data available (training with older model)</p>
        </div>
      </div>
    );
  }

  // Project KPI: ±30m @ 68% confidence = 0.03 km
  const projectKpiKm = 0.03;

  return (
    <div className="card">
      <div className="card-body">
        <h5 className="card-title mb-1">Distance Error Percentiles</h5>
        <p className="text-muted small mb-3">
          68th percentile (p68) is the <strong>project KPI</strong>. Goal: &lt;30m
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={metricsWithPercentiles}>
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
              formatter={(value: number) => `${value.toFixed(3)} km (${(value * 1000).toFixed(0)}m)`}
            />
            <Legend />
            
            {/* Project KPI reference line */}
            <ReferenceLine 
              y={projectKpiKm} 
              stroke="#10b981" 
              strokeDasharray="5 5" 
              label={{ value: 'KPI Target (30m)', position: 'right', fill: '#10b981', fontSize: 12 }}
            />
            
            <Line 
              type="monotone" 
              dataKey="val_distance_p50_km" 
              stroke="#8b5cf6" 
              strokeWidth={2}
              name="p50 (Median)"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line 
              type="monotone" 
              dataKey="val_distance_p68_km" 
              stroke="#f59e0b" 
              strokeWidth={3}
              name="p68 (PROJECT KPI)"
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
            <Line 
              type="monotone" 
              dataKey="val_distance_p95_km" 
              stroke="#ef4444" 
              strokeWidth={2}
              name="p95 (Worst-case)"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

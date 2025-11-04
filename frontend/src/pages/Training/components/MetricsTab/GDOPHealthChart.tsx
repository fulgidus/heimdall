/**
 * GDOPHealthChart Component
 * 
 * Displays GDOP (Geometric Dilution of Precision) metrics over epochs
 * GDOP < 5 is considered "good geometry" for localization
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

interface GDOPHealthChartProps {
  metrics: TrainingMetric[];
}

export const GDOPHealthChart: React.FC<GDOPHealthChartProps> = ({ metrics }) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No GDOP data available</p>
        </div>
      </div>
    );
  }

  // Filter metrics that have GDOP fields
  const metricsWithGDOP = metrics.filter(m => m.mean_gdop !== undefined);

  if (metricsWithGDOP.length === 0) {
    return (
      <div className="card text-center py-5">
        <div className="card-body">
          <p className="text-muted small mb-0">No GDOP data available (training with older model)</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-body">
        <h5 className="card-title mb-1">GDOP Health</h5>
        <p className="text-muted small mb-3">
          Geometric quality of receiver positions. GDOP &lt;5 = good geometry
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={metricsWithGDOP}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis 
              dataKey="epoch" 
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
              stroke="#6b7280"
            />
            <YAxis 
              yAxisId="left"
              label={{ value: 'Mean GDOP', angle: -90, position: 'insideLeft' }}
              stroke="#6b7280"
              domain={[0, 'auto']}
            />
            <YAxis 
              yAxisId="right"
              orientation="right"
              label={{ value: 'Good Geometry %', angle: 90, position: 'insideRight' }}
              stroke="#10b981"
              domain={[0, 100]}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#fff', 
                border: '1px solid #e5e7eb',
                borderRadius: '0.375rem',
              }}
              formatter={(value: number, name: string) => {
                if (name === '% GDOP<5') return `${value.toFixed(1)}%`;
                return value.toFixed(2);
              }}
            />
            <Legend />
            
            {/* GDOP threshold reference */}
            <ReferenceLine 
              yAxisId="left"
              y={5} 
              stroke="#f59e0b" 
              strokeDasharray="5 5" 
              label={{ value: 'Good Geometry Threshold', position: 'right', fill: '#f59e0b', fontSize: 11 }}
            />
            
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="mean_gdop" 
              stroke="#3b82f6" 
              strokeWidth={2}
              name="Mean GDOP"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="gdop_below_5_percent" 
              stroke="#10b981" 
              strokeWidth={2}
              name="% GDOP<5"
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="mt-2">
          <small className="text-muted">
            💡 Higher % GDOP&lt;5 means more samples with reliable receiver geometry
          </small>
        </div>
      </div>
    </div>
  );
};

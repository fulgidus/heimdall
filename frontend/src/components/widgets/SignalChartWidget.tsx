import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
} from 'chart.js';
import { useDashboardStore } from '@/store';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

interface SignalChartWidgetProps {
    widgetId: string;
}

interface DataPoint {
    time: string;
    count: number;
}

export const SignalChartWidget: React.FC<SignalChartWidgetProps> = () => {
    const { metrics } = useDashboardStore();
    const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
    const maxDataPoints = 20;

    useEffect(() => {
        // Add new data point every 5 seconds
        const interval = setInterval(() => {
            const now = new Date();
            const timeString = now.toLocaleTimeString('en-US', { 
                hour12: false, 
                hour: '2-digit', 
                minute: '2-digit',
                second: '2-digit'
            });

            setDataPoints(prev => {
                const newPoints = [
                    ...prev,
                    {
                        time: timeString,
                        count: metrics.signalDetections || 0,
                    },
                ].slice(-maxDataPoints);

                return newPoints;
            });
        }, 5000);

        return () => clearInterval(interval);
    }, [metrics.signalDetections]);

    const chartData = {
        labels: dataPoints.map(d => d.time),
        datasets: [
            {
                label: 'Signal Detections',
                data: dataPoints.map(d => d.count),
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: true,
                tension: 0.4,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
            title: {
                display: false,
            },
        },
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    precision: 0,
                },
            },
            x: {
                ticks: {
                    maxRotation: 45,
                    minRotation: 45,
                },
            },
        },
    };

    return (
        <div className="widget-content">
            <div className="d-flex align-items-center justify-content-between mb-3">
                <div>
                    <h3 className="h4 mb-0">{metrics.signalDetections || 0}</h3>
                    <p className="text-muted small mb-0">Total Detections (24h)</p>
                </div>
                <i className="ph ph-chart-line fs-1 text-primary" />
            </div>
            
            <div style={{ height: '200px' }}>
                {dataPoints.length > 0 ? (
                    <Line data={chartData} options={options} />
                ) : (
                    <div className="d-flex align-items-center justify-content-center h-100 text-muted">
                        <div className="text-center">
                            <i className="ph ph-chart-line fs-1 mb-2" />
                            <p className="mb-0">Collecting data...</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

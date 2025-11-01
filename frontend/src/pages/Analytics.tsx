import React, { useEffect, useState } from 'react';
import { Line, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { useDashboardStore, useWebSDRStore, useAnalyticsStore } from '../store';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const Analytics: React.FC = () => {
  const { fetchDashboardData } = useDashboardStore();
  const { websdrs, healthStatus } = useWebSDRStore();
  const {
    predictionMetrics,
    websdrPerformance,
    systemPerformance,
    accuracyDistribution,
    isLoading: analyticsLoading,
    error: analyticsError,
    timeRange,
    setTimeRange,
    fetchAllAnalytics,
    refreshData,
  } = useAnalyticsStore();
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    fetchDashboardData();
    fetchAllAnalytics();
    const interval = setInterval(() => {
      fetchDashboardData();
      fetchAllAnalytics();
    }, 30000);
    return () => clearInterval(interval);
  }, [fetchDashboardData, fetchAllAnalytics]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([fetchDashboardData(), refreshData()]);
    setIsRefreshing(false);
  };

  // Calculate metrics from real analytics data
  const onlineWebSDRs = Object.values(healthStatus).filter(h => h.status === 'online').length;
  const totalWebSDRs = websdrs.length || 7;

  // Calculate totals from time series data
  const totalPredictions = predictionMetrics?.total_predictions?.slice(-1)[0]?.value || 0;
  const successfulPredictions = predictionMetrics?.successful_predictions?.slice(-1)[0]?.value || 0;
  const avgUncertainty = predictionMetrics?.average_uncertainty?.slice(-1)[0]?.value || 0;

  const avgAccuracy = avgUncertainty > 0 ? `±${avgUncertainty.toFixed(1)}m` : 'N/A';
  const successRate =
    totalPredictions > 0 ? ((successfulPredictions / totalPredictions) * 100).toFixed(1) : '0';

  return (
    <>
      {/* Breadcrumb */}
      <div className="page-header">
        <div className="page-block">
          <div className="row align-items-center">
            <div className="col-md-12">
              <ul className="breadcrumb">
                <li className="breadcrumb-item">
                  <a href="/dashboard">Home</a>
                </li>
                <li className="breadcrumb-item" aria-current="page">
                  Analytics
                </li>
              </ul>
            </div>
            <div className="col-md-12">
              <div className="page-header-title">
                <h2 className="mb-0">Analytics</h2>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Loading/Error States */}
      {analyticsLoading && (
        <div className="row mb-3">
          <div className="col-12">
            <div className="alert alert-info">
              <i className="ph ph-spinner ph-spin me-2"></i>
              Loading analytics data...
            </div>
          </div>
        </div>
      )}

      {analyticsError && (
        <div className="row mb-3">
          <div className="col-12">
            <div className="alert alert-danger">
              <i className="ph ph-warning-circle me-2"></i>
              Failed to load analytics data: {analyticsError}
            </div>
          </div>
        </div>
      )}

      {/* Time Range Selector */}
      <div className="row mb-3">
        <div className="col-12">
          <div className="btn-group" role="group">
            <button
              type="button"
              className={`btn ${timeRange === '24h' ? 'btn-primary' : 'btn-outline-primary'}`}
              onClick={() => {
                setTimeRange('24h');
                fetchAllAnalytics('24h');
              }}
            >
              24 Hours
            </button>
            <button
              type="button"
              className={`btn ${timeRange === '7d' ? 'btn-primary' : 'btn-outline-primary'}`}
              onClick={() => {
                setTimeRange('7d');
                fetchAllAnalytics('7d');
              }}
            >
              7 Days
            </button>
            <button
              type="button"
              className={`btn ${timeRange === '30d' ? 'btn-primary' : 'btn-outline-primary'}`}
              onClick={() => {
                setTimeRange('30d');
                fetchAllAnalytics('30d');
              }}
            >
              30 Days
            </button>
            <button
              type="button"
              className="btn btn-outline-secondary ms-auto"
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              <i className={`ph ph-arrows-clockwise ${isRefreshing ? 'spin' : ''}`}></i>
              {isRefreshing ? ' Refreshing...' : ' Refresh'}
            </button>
          </div>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="row">
        {/* Total Localizations */}
        <div className="col-md-6 col-xl-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="flex-shrink-0">
                  <div className="avtar avtar-s bg-light-primary rounded">
                    <i className="ph ph-chart-line f-20"></i>
                  </div>
                </div>
                <div className="flex-grow-1 ms-3">
                  <h6 className="mb-0">Total Predictions</h6>
                  <h4 className="mb-0">{totalPredictions.toLocaleString()}</h4>
                </div>
              </div>
              <div className="bg-body p-2 mt-2 rounded">
                <div className="d-flex align-items-center justify-content-between">
                  <p className="text-muted mb-0 f-12">Success Rate</p>
                  <span className="badge bg-light-success">{successRate}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Model Accuracy */}
        <div className="col-md-6 col-xl-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="flex-shrink-0">
                  <div className="avtar avtar-s bg-light-success rounded">
                    <i className="ph ph-target f-20"></i>
                  </div>
                </div>
                <div className="flex-grow-1 ms-3">
                  <h6 className="mb-0">Model Accuracy</h6>
                  <h4 className="mb-0">{avgAccuracy}</h4>
                </div>
              </div>
              <div className="bg-body p-2 mt-2 rounded">
                <div className="d-flex align-items-center justify-content-between">
                  <p className="text-muted mb-0 f-12">Confidence</p>
                  <span className="badge bg-light-success">High</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Active Receivers */}
        <div className="col-md-6 col-xl-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="flex-shrink-0">
                  <div className="avtar avtar-s bg-light-warning rounded">
                    <i className="ph ph-radio-button f-20"></i>
                  </div>
                </div>
                <div className="flex-grow-1 ms-3">
                  <h6 className="mb-0">Active Receivers</h6>
                  <h4 className="mb-0">
                    {onlineWebSDRs}/{totalWebSDRs}
                  </h4>
                </div>
              </div>
              <div className="bg-body p-2 mt-2 rounded">
                <div className="d-flex align-items-center justify-content-between">
                  <p className="text-muted mb-0 f-12">Uptime</p>
                  <span className="badge bg-light-warning">
                    {Math.round((onlineWebSDRs / totalWebSDRs) * 100)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* System Uptime */}
        <div className="col-md-6 col-xl-3">
          <div className="card">
            <div className="card-body">
              <div className="d-flex align-items-center">
                <div className="flex-shrink-0">
                  <div className="avtar avtar-s bg-light-info rounded">
                    <i className="ph ph-activity f-20"></i>
                  </div>
                </div>
                <div className="flex-grow-1 ms-3">
                  <h6 className="mb-0">System Uptime</h6>
                  <h4 className="mb-0">
                    {systemPerformance?.cpu_usage?.slice(-1)[0]?.value
                      ? `${systemPerformance.cpu_usage.slice(-1)[0].value.toFixed(1)}%`
                      : '0%'}
                  </h4>
                </div>
              </div>
              <div className="bg-body p-2 mt-2 rounded">
                <div className="d-flex align-items-center justify-content-between">
                  <p className="text-muted mb-0 f-12">Status</p>
                  <span className="badge bg-light-success">Online</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="row">
        {/* Prediction Trends Chart */}
        <div className="col-lg-8">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Prediction Trends</h5>
            </div>
            <div className="card-body">
              {predictionMetrics ? (
                <Line
                  data={{
                    labels: predictionMetrics.total_predictions.map(p =>
                      new Date(p.timestamp).toLocaleDateString()
                    ),
                    datasets: [
                      {
                        label: 'Total Predictions',
                        data: predictionMetrics.total_predictions.map(p => p.value),
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1,
                      },
                      {
                        label: 'Successful Predictions',
                        data: predictionMetrics.successful_predictions.map(p => p.value),
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.1,
                      },
                      {
                        label: 'Failed Predictions',
                        data: predictionMetrics.failed_predictions.map(p => p.value),
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    plugins: {
                      legend: {
                        position: 'top' as const,
                      },
                      title: {
                        display: true,
                        text: 'Prediction Trends Over Time',
                      },
                    },
                    scales: {
                      y: {
                        beginAtZero: true,
                      },
                    },
                  }}
                />
              ) : (
                <div className="text-center py-5">
                  <i className="ph ph-chart-line-up f-40 text-primary mb-3"></i>
                  <p className="text-muted">Loading prediction trends data...</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Accuracy Distribution */}
        <div className="col-lg-4">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Accuracy Distribution</h5>
            </div>
            <div className="card-body">
              {accuracyDistribution ? (
                <div>
                  <Pie
                    data={{
                      labels: accuracyDistribution.accuracy_ranges,
                      datasets: [
                        {
                          data: accuracyDistribution.counts,
                          backgroundColor: [
                            'rgba(75, 192, 192, 0.8)',
                            'rgba(54, 162, 235, 0.8)',
                            'rgba(255, 206, 86, 0.8)',
                            'rgba(255, 99, 132, 0.8)',
                            'rgba(153, 102, 255, 0.8)',
                          ],
                          borderColor: [
                            'rgba(75, 192, 192, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(255, 99, 132, 1)',
                            'rgba(153, 102, 255, 1)',
                          ],
                          borderWidth: 1,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: {
                          position: 'bottom' as const,
                        },
                      },
                    }}
                  />
                  <div className="mt-3 text-center">
                    <h6 className="mb-0">{avgAccuracy}</h6>
                    <p className="text-muted f-12 mb-0 mt-2">Average Accuracy</p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-4">
                  <div className="mb-3">
                    <i className="ph ph-chart-pie-slice f-40 text-success"></i>
                  </div>
                  <h3 className="mb-0">{avgAccuracy}</h3>
                  <p className="text-muted f-12 mb-0 mt-2">Average Accuracy</p>
                  <p className="text-muted mt-2">Loading distribution data...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* WebSDR Performance */}
      <div className="row">
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">WebSDR Performance</h5>
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-hover mb-0">
                  <thead>
                    <tr>
                      <th>Receiver</th>
                      <th>Uptime</th>
                      <th>Avg SNR</th>
                      <th>Total Acquisitions</th>
                      <th>Success Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {websdrPerformance && websdrPerformance.length > 0
                      ? websdrPerformance.map(sdr => {
                          const successRate =
                            sdr.total_acquisitions > 0
                              ? (
                                  (sdr.successful_acquisitions / sdr.total_acquisitions) *
                                  100
                                ).toFixed(1)
                              : '0';

                          return (
                            <tr key={sdr.websdr_id}>
                              <td>
                                <div className="d-flex align-items-center">
                                  <div
                                    className={`avtar avtar-xs ${sdr.uptime_percentage > 80 ? 'bg-light-success' : 'bg-light-warning'}`}
                                  >
                                    <i className="ph ph-radio-button"></i>
                                  </div>
                                  <span className="ms-2">{sdr.name}</span>
                                </div>
                              </td>
                              <td>{sdr.uptime_percentage.toFixed(1)}%</td>
                              <td>{sdr.average_snr.toFixed(1)} dB</td>
                              <td>{sdr.total_acquisitions}</td>
                              <td>
                                <span
                                  className={`badge ${parseFloat(successRate) > 80 ? 'bg-light-success' : 'bg-light-warning'}`}
                                >
                                  {successRate}%
                                </span>
                              </td>
                            </tr>
                          );
                        })
                      : websdrs.map(sdr => {
                          const health = healthStatus[sdr.id];
                          const isOnline = health?.status === 'online';
                          const responseTime = health?.response_time_ms || 0;
                          const reliability = isOnline ? Math.max(80, 100 - responseTime / 10) : 0;

                          return (
                            <tr key={sdr.id}>
                              <td>
                                <div className="d-flex align-items-center">
                                  <div
                                    className={`avtar avtar-xs ${isOnline ? 'bg-light-success' : 'bg-light-danger'}`}
                                  >
                                    <i className="ph ph-radio-button"></i>
                                  </div>
                                  <span className="ms-2">{sdr.name}</span>
                                </div>
                              </td>
                              <td>{isOnline ? 'Online' : 'Offline'}</td>
                              <td>-</td>
                              <td>-</td>
                              <td>
                                <div className="d-flex align-items-center">
                                  <div
                                    className="progress flex-grow-1 me-2"
                                    style={{ height: '6px' }}
                                  >
                                    <div
                                      className={`progress-bar ${isOnline ? 'bg-success' : 'bg-danger'}`}
                                      role="progressbar"
                                      style={{ width: `${reliability}%` }}
                                    ></div>
                                  </div>
                                  <span className="f-12">{Math.round(reliability)}%</span>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                    {(!websdrPerformance || websdrPerformance.length === 0) &&
                      websdrs.length === 0 && (
                        <tr>
                          <td colSpan={5} className="text-center py-4">
                            <i className="ph ph-warning-circle f-40 text-muted mb-2"></i>
                            <p className="text-muted mb-0">No WebSDR data available</p>
                          </td>
                        </tr>
                      )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Analytics;

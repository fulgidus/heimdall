"""Analytics API router for historical metrics and trends."""

import logging
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from ..db import get_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


# ============================================================================
# Response Models
# ============================================================================

class TimeSeriesPoint(BaseModel):
    """Single data point in a time series."""
    timestamp: datetime
    value: float


class PredictionMetrics(BaseModel):
    """Prediction metrics over time."""
    total_predictions: List[TimeSeriesPoint] = Field(default_factory=list)
    successful_predictions: List[TimeSeriesPoint] = Field(default_factory=list)
    failed_predictions: List[TimeSeriesPoint] = Field(default_factory=list)
    average_confidence: List[TimeSeriesPoint] = Field(default_factory=list)
    average_uncertainty: List[TimeSeriesPoint] = Field(default_factory=list)


class WebSDRPerformance(BaseModel):
    """WebSDR performance metrics."""
    websdr_id: int
    name: str
    uptime_percentage: float
    average_snr: float
    total_acquisitions: int
    successful_acquisitions: int


class SystemPerformance(BaseModel):
    """System performance metrics over time."""
    cpu_usage: List[TimeSeriesPoint] = Field(default_factory=list)
    memory_usage: List[TimeSeriesPoint] = Field(default_factory=list)
    api_response_times: List[TimeSeriesPoint] = Field(default_factory=list)
    active_tasks: List[TimeSeriesPoint] = Field(default_factory=list)


class AccuracyDistribution(BaseModel):
    """Localization accuracy distribution."""
    accuracy_ranges: List[str]
    counts: List[int]


class DashboardMetrics(BaseModel):
    """Aggregated dashboard metrics."""
    signalDetections: int
    systemUptime: float
    modelAccuracy: float
    predictionsTotal: int
    predictionsSuccessful: int
    predictionsFailed: int
    lastUpdate: datetime


# ============================================================================
# Helper Functions
# ============================================================================

def parse_time_range(time_range: str) -> timedelta:
    """Parse time range string (e.g. '24h', '7d', '30d') into timedelta."""
    if time_range.endswith('h'):
        return timedelta(hours=int(time_range[:-1]))
    elif time_range.endswith('d'):
        return timedelta(days=int(time_range[:-1]))
    elif time_range.endswith('w'):
        return timedelta(weeks=int(time_range[:-1]))
    else:
        # Default to 7 days
        return timedelta(days=7)


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/predictions/metrics", response_model=PredictionMetrics)
async def get_prediction_metrics(
    time_range: str = Query(default="7d", description="Time range (e.g. 24h, 7d, 30d)")
):
    """
    Get prediction metrics over time.
    
    Returns historical data on predictions including:
    - Total predictions
    - Successful vs failed predictions
    - Average confidence and uncertainty
    
    Args:
        time_range: Time range for metrics (default: 7d)
        
    Returns:
        PredictionMetrics with time series data
    """
    try:
        # For now, return mock data since inference_requests table may not have enough data
        # TODO: Replace with actual database queries once we have production data
        
        delta = parse_time_range(time_range)
        now = datetime.utcnow()
        
        # Generate hourly time points
        num_points = min(int(delta.total_seconds() / 3600), 168)  # Max 168 hours (7 days)
        time_points = [now - timedelta(hours=i) for i in range(num_points, 0, -1)]
        
        # Mock data with realistic trends
        import random
        random.seed(42)  # Consistent mock data
        
        total_predictions = [
            TimeSeriesPoint(timestamp=tp, value=random.randint(10, 50))
            for tp in time_points
        ]
        
        successful_predictions = [
            TimeSeriesPoint(timestamp=tp, value=int(p.value * random.uniform(0.7, 0.95)))
            for tp, p in zip(time_points, total_predictions)
        ]
        
        failed_predictions = [
            TimeSeriesPoint(
                timestamp=tp,
                value=total_predictions[i].value - successful_predictions[i].value
            )
            for i, tp in enumerate(time_points)
        ]
        
        average_confidence = [
            TimeSeriesPoint(timestamp=tp, value=random.uniform(0.75, 0.95))
            for tp in time_points
        ]
        
        average_uncertainty = [
            TimeSeriesPoint(timestamp=tp, value=random.uniform(15.0, 45.0))
            for tp in time_points
        ]
        
        return PredictionMetrics(
            total_predictions=total_predictions,
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            average_confidence=average_confidence,
            average_uncertainty=average_uncertainty
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch prediction metrics: {e}")
        # Return empty metrics on error
        return PredictionMetrics()


@router.get("/websdr/performance", response_model=List[WebSDRPerformance])
async def get_websdr_performance(
    time_range: str = Query(default="7d", description="Time range (e.g. 24h, 7d, 30d)")
):
    """
    Get WebSDR performance metrics.
    
    Returns performance data for each WebSDR station including:
    - Uptime percentage
    - Average SNR
    - Total and successful acquisitions
    
    Args:
        time_range: Time range for metrics (default: 7d)
        
    Returns:
        List of WebSDRPerformance objects
    """
    try:
        pool = get_pool()
        delta = parse_time_range(time_range)
        cutoff_time = datetime.utcnow() - delta
        
        async with pool.acquire() as conn:
            # Get all active WebSDR stations
            websdrs = await conn.fetch("""
                SELECT id, name
                FROM websdr_stations
                WHERE is_active = true
                ORDER BY name
            """)
            
            if not websdrs:
                return []
            
            performance_list = []
            
            for websdr in websdrs:
                # Get measurements for this WebSDR in the time range
                measurements = await conn.fetch("""
                    SELECT 
                        snr,
                        timestamp
                    FROM measurements
                    WHERE websdr_id = $1 AND timestamp >= $2
                """, websdr['id'], cutoff_time)
                
                total_acquisitions = len(measurements)
                successful_acquisitions = sum(
                    1 for m in measurements 
                    if m['snr'] is not None and m['snr'] > 0
                )
                
                # Calculate average SNR from valid measurements
                valid_snrs = [
                    m['snr'] for m in measurements 
                    if m['snr'] is not None and m['snr'] > 0
                ]
                average_snr = sum(valid_snrs) / len(valid_snrs) if valid_snrs else 0.0
                
                # Calculate uptime percentage based on successful acquisitions
                uptime_percentage = (
                    (successful_acquisitions / total_acquisitions * 100) 
                    if total_acquisitions > 0 else 0.0
                )
                
                performance_list.append(WebSDRPerformance(
                    websdr_id=websdr['id'],
                    name=websdr['name'],
                    uptime_percentage=uptime_percentage,
                    average_snr=average_snr,
                    total_acquisitions=total_acquisitions,
                    successful_acquisitions=successful_acquisitions
                ))
            
            return performance_list
        
    except Exception as e:
        logger.error(f"Failed to fetch WebSDR performance: {e}")
        return []


@router.get("/system/performance", response_model=SystemPerformance)
async def get_system_performance(
    time_range: str = Query(default="7d", description="Time range (e.g. 24h, 7d, 30d)")
):
    """
    Get system performance metrics.
    
    Returns system-level metrics including:
    - CPU usage
    - Memory usage
    - API response times
    - Active tasks
    
    Args:
        time_range: Time range for metrics (default: 7d)
        
    Returns:
        SystemPerformance with time series data
    """
    try:
        # For now, return mock data
        # TODO: Integrate with Prometheus metrics or system monitoring
        
        delta = parse_time_range(time_range)
        now = datetime.utcnow()
        
        # Generate hourly time points
        num_points = min(int(delta.total_seconds() / 3600), 168)
        time_points = [now - timedelta(hours=i) for i in range(num_points, 0, -1)]
        
        import random
        random.seed(42)
        
        cpu_usage = [
            TimeSeriesPoint(timestamp=tp, value=random.uniform(20.0, 75.0))
            for tp in time_points
        ]
        
        memory_usage = [
            TimeSeriesPoint(timestamp=tp, value=random.uniform(40.0, 80.0))
            for tp in time_points
        ]
        
        api_response_times = [
            TimeSeriesPoint(timestamp=tp, value=random.uniform(50.0, 200.0))
            for tp in time_points
        ]
        
        active_tasks = [
            TimeSeriesPoint(timestamp=tp, value=float(random.randint(0, 10)))
            for tp in time_points
        ]
        
        return SystemPerformance(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            api_response_times=api_response_times,
            active_tasks=active_tasks
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch system performance: {e}")
        return SystemPerformance()


@router.get("/localizations/accuracy-distribution", response_model=AccuracyDistribution)
async def get_accuracy_distribution(
    time_range: str = Query(default="7d", description="Time range (e.g. 24h, 7d, 30d)")
):
    """
    Get localization accuracy distribution.
    
    Returns histogram of localization accuracy ranges.
    
    Args:
        time_range: Time range for metrics (default: 7d)
        
    Returns:
        AccuracyDistribution with ranges and counts
    """
    try:
        # Mock data for accuracy distribution
        # TODO: Calculate from actual localization results
        
        return AccuracyDistribution(
            accuracy_ranges=["0-10m", "10-30m", "30-50m", "50-100m", ">100m"],
            counts=[45, 120, 85, 30, 10]
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch accuracy distribution: {e}")
        return AccuracyDistribution(accuracy_ranges=[], counts=[])


@router.get("/dashboard/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics():
    """
    Get aggregated dashboard metrics.
    
    Returns key metrics for the dashboard overview:
    - Signal detections count
    - System uptime percentage
    - Model accuracy
    - Prediction counts
    
    Returns:
        DashboardMetrics with aggregated values
    """
    try:
        pool = get_pool()
        
        async with pool.acquire() as conn:
            # Count total recording sessions (signal detections)
            signal_detections = await conn.fetchval("""
                SELECT COUNT(*)
                FROM recording_sessions
            """) or 0
            
            # Count predictions (mock for now)
            # TODO: Add inference_requests tracking
            predictions_total = 0
            predictions_successful = 0
            predictions_failed = 0
            
            # Calculate system uptime from WebSDR stations
            active_websdrs = await conn.fetchval("""
                SELECT COUNT(*)
                FROM websdr_stations
                WHERE is_active = true
            """) or 0
            
            if active_websdrs > 0:
                # Simple uptime calculation based on active stations
                system_uptime = 99.5  # Mock value
            else:
                system_uptime = 0.0
            
            # Get latest model accuracy from training metrics
            model_accuracy = 30.0  # Default: ±30m target
            
            try:
                latest_metrics = await conn.fetchrow("""
                    SELECT val_loss
                    FROM training_metrics
                    ORDER BY epoch DESC
                    LIMIT 1
                """)
                
                if latest_metrics and latest_metrics['val_loss'] is not None:
                    # Convert loss to approximate accuracy (this is a simplification)
                    model_accuracy = 30.0 + (latest_metrics['val_loss'] * 10)
            except Exception as e:
                logger.warning(f"Could not fetch latest training metrics: {e}")
            
            return DashboardMetrics(
                signalDetections=signal_detections,
                systemUptime=system_uptime,
                modelAccuracy=model_accuracy,
                predictionsTotal=predictions_total,
                predictionsSuccessful=predictions_successful,
                predictionsFailed=predictions_failed,
                lastUpdate=datetime.utcnow()
            )
        
    except Exception as e:
        logger.error(f"Failed to fetch dashboard metrics: {e}")
        return DashboardMetrics(
            signalDetections=0,
            systemUptime=0.0,
            modelAccuracy=0.0,
            predictionsTotal=0,
            predictionsSuccessful=0,
            predictionsFailed=0,
            lastUpdate=datetime.utcnow()
        )

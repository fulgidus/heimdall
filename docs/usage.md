# Usage Guide

## Getting Started

Welcome to Heimdall! This guide will help you use the system to locate radio transmissions.

### Basic Workflow

1. **Configure WebSDR Network**
   - Ensure WebSDR stations are online
   - Configure frequencies of interest
   - Set up signal detection parameters

2. **Start RF Acquisition**
   - Initiate data collection from WebSDR receivers
   - Monitor signal quality from each station
   - Verify data ingestion into the pipeline

3. **Process Signals**
   - Extract features from collected signals
   - Run anomaly detection models
   - Generate predictions

4. **View Results**
   - Access the web dashboard
   - Review localization results
   - Analyze uncertainty estimates

## API Usage

### REST API Endpoints

#### Submit RF Acquisition Task

```bash
curl -X POST http://localhost:8000/api/v1/rf-tasks \
  -H "Content-Type: application/json" \
  -d '{
    "frequencies": [145.500, 433.025],
    "duration": 60,
    "bandwidth": 2400
  }'
```

#### Get Task Status

```bash
curl http://localhost:8000/api/v1/rf-tasks/{task_id}
```

#### Get Localization Results

```bash
curl http://localhost:8000/api/v1/results/{task_id}
```

### Python Client

```python
from heimdall_client import HeimdallClient

client = HeimdallClient("http://localhost:8000")

# Submit task
task = client.submit_rf_acquisition(
    frequencies=[145.500, 433.025],
    duration=60
)

# Poll for results
result = client.wait_for_result(task.id, timeout=300)

# Access results
print(f"Latitude: {result.location.latitude}")
print(f"Longitude: {result.location.longitude}")
print(f"Uncertainty: {result.location.uncertainty_m}m")
```

## Web Dashboard

The web dashboard is available at `http://localhost:3000`

### Features

- **Real-time Map**: View localization results on interactive map
- **Task Management**: Submit and monitor RF acquisition tasks
- **Historical Analysis**: Review past acquisitions and results
- **Performance Metrics**: Monitor system performance and health
- **Configuration**: Adjust WebSDR and processing parameters

### Navigation

- **Dashboard**: Overview of recent results
- **Tasks**: View all active and completed tasks
- **Map**: Interactive map with localization results
- **Settings**: Configure system parameters
- **Help**: Documentation and troubleshooting

## Common Tasks

### Locating a Specific Transmission

```python
# Submit acquisition on specific frequency
task = client.submit_rf_acquisition(
    frequencies=[145.500],  # Amateur 2m frequency
    duration=120  # 2 minute acquisition
)

# Wait for completion
result = client.wait_for_result(task.id)

# Display results
print(f"Signal found at: {result.location}")
print(f"Confidence: {result.confidence}")
```

### Batch Processing Multiple Frequencies

```python
# Submit multiple frequencies
frequencies = [145.500, 145.550, 433.025, 433.075]
tasks = []

for freq in frequencies:
    task = client.submit_rf_acquisition(frequencies=[freq])
    tasks.append(task)

# Collect results
results = [client.wait_for_result(t.id) for t in tasks]

# Process results
for result in results:
    print(f"{result.frequency}: {result.location}")
```

### Setting Up Continuous Monitoring

```python
# Schedule periodic acquisitions
from heimdall_client import SchedulerClient

scheduler = SchedulerClient("http://localhost:8000")

# Every 5 minutes on 145.500 MHz
scheduler.schedule_acquisition(
    frequencies=[145.500],
    interval=300,  # seconds
    duration=60
)

# Subscribe to results
def on_result(result):
    print(f"New result: {result.location}")

scheduler.on_result = on_result
scheduler.start()
```

## Performance Tuning

### Optimizing Accuracy

- **Increase acquisition duration**: Longer signals provide more data
- **Use multiple frequencies**: Broader spectrum analysis improves accuracy
- **Verify station geometry**: Better baseline = better triangulation

### Improving Speed

- **Reduce bandwidth**: Fewer Hz per sample = faster processing
- **Use GPU inference**: Enable ONNX model for 1.5-2.5x speedup
- **Parallel processing**: Process multiple frequencies simultaneously

## Troubleshooting

### No Results Returned

```bash
# Check task status
curl http://localhost:8000/api/v1/rf-tasks/{task_id}

# Check service logs
docker-compose logs ml-detector

# Verify WebSDR connectivity
curl http://localhost:8000/api/v1/health/websdrs
```

### High Uncertainty

- **Weak signal**: Try longer acquisition duration
- **Poor geometry**: Signal from edge of network coverage
- **Interference**: Use narrower bandwidth to filter noise

### Processing Timeout

- **Reduce duration**: 60s acquisitions are faster than 120s
- **Scale infrastructure**: Add more worker containers

---

**Next Steps**: [API Reference](./api_reference.md) | [Troubleshooting](./troubleshooting_guide.md)

# Frequently Asked Questions

## Installation & Setup

### Q: Do I need all the required software installed?

**A**: For development, yes. For production, Docker handles most dependencies. See [Installation Guide](./installation.md) for minimal requirements.

### Q: How do I configure WebSDR stations?

**A**: Edit the WebSDR configuration in the database or through the API. See [WebSDR Configuration](./websdrs.md) for details.

### Q: What if Docker containers fail to start?

**A**: Check logs with `docker-compose logs`. Common issues:
- Port conflicts: Change port mappings in `.env`
- Memory issues: Increase Docker memory allocation
- Network issues: Ensure all services can reach database

## Usage & API

### Q: How do I submit an RF acquisition task?

**A**: Use the REST API:
```bash
curl -X POST http://localhost:8000/api/v1/rf-tasks \
  -H "Content-Type: application/json" \
  -d '{"frequencies": [145.500], "duration": 60}'
```

### Q: What's the typical latency?

**A**: 
- API response: ~52ms
- RF acquisition: 63-70s (network-bound)
- Processing: <500ms
- Total: ~2-3 minutes for complete pipeline

### Q: How many concurrent tasks can I submit?

**A**: System handles 50+ concurrent RF acquisitions. Scale workers as needed.

## Performance & Optimization

### Q: How do I improve localization accuracy?

**A**: 
- Increase acquisition duration (longer signals = more data)
- Use better WebSDR stations (higher SNR)
- Verify station geometry (triangulation works best with good baseline)
- Check signal strength (-90 dBm minimum)

### Q: How do I speed up processing?

**A**:
- Use GPU inference (1.5-2.5x speedup)
- Enable model quantization
- Reduce bandwidth for faster processing
- Run workers in parallel

### Q: What's the GPU requirement?

**A**: 6-8GB VRAM minimum. Works on NVIDIA (CUDA), AMD (ROCm), and Apple (MPS) GPUs.

## Troubleshooting

### Q: My tasks are timing out

**A**: 
- Check WebSDR availability
- Increase timeout in config
- Scale infrastructure (add workers)
- Check network connectivity

### Q: Inference results have high uncertainty

**A**:
- Signal too weak: Increase acquisition duration
- Poor station geometry: Wait for better coverage
- Too much interference: Use narrower bandwidth

### Q: How do I view service logs?

**A**: 
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ml-detector

# Last N lines
docker-compose logs --tail 100 api-gateway
```

## Development

### Q: How do I set up a development environment?

**A**: See [Installation Guide](./installation.md) for detailed steps.

### Q: How do I run tests?

**A**:
```bash
# All tests
make test

# With coverage
make test-coverage

# Specific test
pytest tests/test_signal_processor.py -v
```

### Q: What's the test coverage?

**A**: Currently 85%+ coverage. Run `make test-coverage` to see detailed report.

### Q: How do I add a new feature?

**A**: 
1. Create feature branch: `git checkout -b feature/name`
2. Implement feature with tests
3. Submit pull request
4. Update documentation
5. See [Contributing Guidelines](./contributing.md)

## Architecture & Design

### Q: Why PostgreSQL + TimescaleDB?

**A**: TimescaleDB optimizes for time-series data (signals, measurements) with automatic partitioning and compression.

### Q: Why separate WebSDR Collector and Signal Processor?

**A**: Separation of concerns allows independent scaling. Collectors can be IO-bound, processors can be CPU-bound.

### Q: Why use ONNX for inference?

**A**: ONNX models are 1.5-2.5x faster than PyTorch, enabling real-time inference.

### Q: How is the model trained?

**A**: PyTorch Lightning with MLflow tracking. See [Training Guide](./TRAINING.md).

## Deployment

### Q: How do I deploy to production?

**A**: 
1. Build production images: `docker-compose -f docker-compose.prod.yml build`
2. Deploy to Kubernetes: `helm install heimdall ./helm/heimdall`
3. Configure environment: Set production `.env`
4. Monitor with Prometheus/Grafana

### Q: What resources do I need?

**A**: Minimum:
- 2 CPU cores
- 4GB RAM
- 10GB storage
- For GPU: NVIDIA GPU with 6GB+ VRAM

### Q: How do I monitor the system?

**A**: Use:
- Prometheus for metrics
- Grafana for dashboards
- ELK stack for logs
- Kubernetes dashboard for container status

## Contributing & Community

### Q: How do I contribute?

**A**: See [Contributing Guidelines](./contributing.md). Start with small improvements!

### Q: How is my contribution recognized?

**A**: Contributors listed in [Acknowledgements](./acknowledgements.md) and release notes.

### Q: What's the code style?

**A**: 
- Python: PEP 8 + type hints
- JavaScript: ESLint + TypeScript
- See [Contributing Guidelines](./contributing.md)

### Q: How do I report a bug?

**A**: Open a GitHub Issue with:
- Clear title
- Steps to reproduce
- Expected vs actual behavior
- Environment details

## License & Legal

### Q: What's the license?

**A**: [Creative Commons Non-Commercial License](../LICENSE). See file for details.

### Q: Can I use this commercially?

**A**: No, the CC Non-Commercial license restricts commercial use. Contact project owner for commercial licensing.

### Q: What about data privacy?

**A**: See [Security Considerations](./security_considerations.md) for privacy and security details.

---

## Still Have Questions?

- Check [Project Documentation](./index.md)
- Open a [GitHub Issue](https://github.com/fulgidus/heimdall/issues)
- Check [GitHub Discussions](https://github.com/fulgidus/heimdall/discussions)
- Review [API Reference](./api_reference.md)

**Last Updated**: October 2025

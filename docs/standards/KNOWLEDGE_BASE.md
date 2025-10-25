# Knowledge Base & Continuity

Critical knowledge preservation for the Heimdall project.

## Context Preservation Strategy

Each phase must maintain knowledge through:

1. **Checkpoint Documentation**: Every checkpoint includes lessons learned and decisions made
2. **Code Comments**: Critical decisions documented in code
3. **Architecture Decision Records (ADRs)**: Formal documentation of technical choices
4. **Handoff Notes**: Detailed transfer of knowledge between agents/phases

## Critical Knowledge Areas

### WebSDR Integration

**API Quirks and Limitations:**
- Each of the 7 receivers has different API response formats
- Rate limiting varies by receiver (some limit to 10 req/min)
- Connection stability differs (some receivers drop connections after 5 min)
- Some receivers require authentication tokens

**Frequency Offset Corrections:**
- Each receiver has a calibrated frequency offset (documented in WEBSDRS.md)
- Offsets range from -500 Hz to +800 Hz
- Apply correction: `true_frequency = reported_frequency + offset`

**Timing Synchronization:**
- GPS-disciplined oscillators provide <1 µs accuracy
- Non-GPS receivers: ±10 ms accuracy (acceptable for localization)
- Always use server timestamps, not client timestamps

**Connection Reliability Patterns:**
- Retry with exponential backoff: 1s, 2s, 4s, 8s
- Maximum 3 retries before marking receiver as offline
- Circuit breaker pattern after 5 consecutive failures
- Health checks every 60 seconds

### ML Model Architecture

**Neural Network Design Decisions:**

**Why CNN over Transformer:**
- CNNs better for spatial-temporal patterns in spectrograms
- Lower computational cost (inference <500ms requirement)
- Smaller model size (~120 MB vs ~500 MB)
- Proven effectiveness for audio/RF signal processing

**Architecture:**
```
Input: Mel-spectrogram (128 bins × time steps) × N receivers
↓
ResNet-18 backbone (feature extraction)
↓
Spatial attention layer (focus on signal components)
↓
Fully connected layers (position + uncertainty)
↓
Output: [latitude, longitude, sigma_x, sigma_y]
```

**Loss Function Choice:**

**Gaussian Negative Log-Likelihood (NLL) over MSE:**
- Penalizes overconfident predictions
- Learns uncertainty along with position
- Better calibration for safety-critical applications

```python
loss = 0.5 * log(2π * sigma^2) + (y_true - y_pred)^2 / (2 * sigma^2)
```

**Feature Extraction Pipeline:**

**Mel-Spectrogram Parameters:**
- Sample rate: 48 kHz
- FFT size: 2048
- Hop length: 512
- Mel bins: 128
- Frequency range: 0-24 kHz

**Why mel-scale:**
- Matches human auditory perception
- Reduces dimensionality while preserving discriminative features
- Standard in audio/RF classification tasks

**Training Hyperparameters:**

**Optimized through grid search:**
- Learning rate: 1e-3 (Adam optimizer)
- Batch size: 32 (balances GPU memory and convergence)
- Epochs: 100 (with early stopping patience=10)
- Weight decay: 1e-4 (L2 regularization)
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=5)

**Convergence patterns:**
- Training loss stabilizes around epoch 30-40
- Validation loss plateaus around epoch 50-60
- Early stopping usually triggers around epoch 60-70

### Deployment Architecture

**Microservices Communication Patterns:**

**Synchronous (HTTP/REST):**
- Frontend → API Gateway
- API Gateway → Individual services
- Client → Inference service

**Asynchronous (Message Queue):**
- RF Acquisition tasks (Celery + RabbitMQ)
- Training pipeline orchestration
- Long-running operations (>30 seconds)

**Event-Driven:**
- WebSocket for real-time updates (frontend)
- Redis Pub/Sub for service coordination

**Database Schema Evolution Strategy:**

**Migration approach:**
1. Use Alembic for schema versioning
2. Never modify existing columns (add new, deprecate old)
3. Backward-compatible migrations (support N-1 version)
4. Test migrations on staging before production

**TimescaleDB hypertables:**
- `measurements` table partitioned by time (1 week chunks)
- Automatic compression after 30 days
- Retention policy: 1 year

**Caching Layers and Performance Optimizations:**

**Redis Caching Strategy:**
- Inference results: TTL 1 hour (cache key = hash of preprocessed features)
- WebSDR status: TTL 60 seconds
- Session metadata: TTL 5 minutes

**Expected cache hit rates:**
- Inference: >80% (similar signals within hour)
- WebSDR status: >95% (checked frequently)
- Session metadata: >90% (active sessions reused)

**Database Query Optimizations:**
- Index on `(session_id, timestamp)` for time-range queries
- Materialized views for aggregated statistics
- Connection pooling (max 20 connections per service)

**Monitoring and Alerting Configurations:**

**Critical Metrics:**
- Inference latency P95 < 500ms (alert if >1s)
- API error rate < 1% (alert if >5%)
- Database connection pool usage < 80% (alert if >90%)
- Disk usage < 85% (alert if >90%)
- Memory usage < 80% (alert if >90%)

**Service Health:**
- HTTP health checks every 30 seconds
- Unhealthy threshold: 3 consecutive failures
- Readiness probe: DB + Redis + RabbitMQ connectivity

**Logging Strategy:**
- Structured JSON logs (timestamp, level, service, message, context)
- Log levels: ERROR (always), WARN (production), INFO (staging), DEBUG (development)
- Log rotation: daily, keep 30 days
- Centralized logging: Elasticsearch + Kibana (production)

## Handoff Protocols

### Transitioning Between Phases

When transitioning between phases:

1. **Complete all checkpoints in current phase**
   - Document lessons learned
   - Note any deviations from plan
   - Collect performance metrics

2. **Document deviations from original plan**
   - Why was the change necessary?
   - What was the impact?
   - Would you make the same choice again?

3. **Update `.copilot-instructions` with new learnings**
   - Add common pitfalls
   - Update best practices
   - Document workarounds

4. **Brief next agent on critical decisions and constraints**
   - Technical debt accepted
   - Blocked issues
   - Dependencies on external systems

5. **Ensure all artifacts are properly versioned and accessible**
   - Code committed and pushed
   - Documentation updated
   - Test data backed up

6. **Follow mandatory update rules**
   - Update AGENTS.md phase status
   - Update CHANGELOG.md with changes
   - Update README.md if needed

### Session Handoff Template

```markdown
# Phase X Handoff - [Date]

## Completed Work
- Task 1: Description and outcome
- Task 2: Description and outcome

## Blockers
- Issue #123: Waiting on external dependency
- Performance issue in module Y (profiling needed)

## Critical Decisions
- Decision 1: Chose approach A over B because...
- Decision 2: Deferred feature X to Phase Y+1

## Technical Debt
- Quick fix in module Z (needs refactoring)
- Missing tests for edge case W

## Next Steps
- Priority 1: Complete remaining tasks in Phase X
- Priority 2: Address blocker issues
- Priority 3: Refactor technical debt

## Knowledge Transfer
- Key file: `/path/to/critical/file.py` - handles XYZ
- Gotcha: Remember to restart service after config change
- Tip: Use make command for common tasks
```

## Common Pitfalls and Solutions

### WebSDR Integration
**Pitfall:** Assuming all receivers respond within timeout
**Solution:** Always handle timeouts and partial failures gracefully

### ML Training
**Pitfall:** Training on imbalanced data (urban vs rural locations)
**Solution:** Use weighted sampling or data augmentation

### Database
**Pitfall:** Running migrations without backup
**Solution:** Always backup before migrations, test on staging first

### Frontend
**Pitfall:** Hardcoding API URLs
**Solution:** Use environment variables for all configuration

### Docker
**Pitfall:** Building images without layer caching
**Solution:** Order Dockerfile commands from least to most frequently changing

## Version Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11.x | Don't use 3.12 (some deps incompatible) |
| Node.js | 18.x or 20.x | pnpm requires 18+ |
| Docker | 20.10+ | For BuildKit support |
| PostgreSQL | 15.x | For TimescaleDB 2.11+ |
| TimescaleDB | 2.11+ | For continuous aggregates |
| Redis | 7.x | For JSON support |
| RabbitMQ | 3.12+ | For stream support |

## Emergency Procedures

### Service Outage
1. Check health endpoints: `make health-check`
2. Check container logs: `docker-compose logs -f <service>`
3. Restart service: `docker-compose restart <service>`
4. If persistent, rollback: `git revert <commit>` and redeploy

### Database Corruption
1. Stop all services: `docker-compose down`
2. Restore from backup: `pg_restore -d heimdall backup.sql`
3. Verify data integrity: Run test queries
4. Restart services: `docker-compose up -d`

### Model Performance Degradation
1. Check MLflow metrics: Open http://localhost:5000
2. Compare with baseline performance
3. If >10% degradation, rollback to previous model
4. Investigate data drift or model staleness

## Contact Information

**Project Owner:** fulgidus (alessio.corsi@gmail.com)

**For Emergencies:**
- Production outage: Check #alerts channel
- Security issue: Email security@heimdall.org
- Data loss: Immediately stop services and contact owner

## Additional Resources

- [Agent Handoff Protocol](../agents/20251022_080000_handoff_protocol.md)
- [Architecture Guide](../ARCHITECTURE.md)
- [Development Guide](../DEVELOPMENT.md)
- [Project Standards](PROJECT_STANDARDS.md)

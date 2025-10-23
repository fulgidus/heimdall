# 🎯 Phase 1: Infrastructure Deployment Instructions

## Status: ✅ Ready to Deploy

All Phase 1 infrastructure code has been created. Follow these instructions to deploy.

---

## ⚠️ Prerequisites Check

Before starting, ensure you have:

- [x] Docker Desktop installed (`docker --version`)
- [x] Docker Compose v2.0+ (`docker-compose --version`)
- [x] At least 8GB RAM available
- [x] At least 20GB free disk space
- [x] Project files downloaded/cloned

---

## 🚀 Deployment Steps (Quick Version)

### Step 1: Navigate to Project
```powershell
cd c:\Users\aless\Documents\Projects\heimdall
```

### Step 2: Verify Environment File
```powershell
# Check if .env exists
ls .env

# If not exists, create from template:
copy .env.example .env
```

### Step 3: Start Infrastructure
```powershell
# Start all services in background
docker-compose up -d

# Watch the startup process
docker-compose logs -f
```

**⏱️ Expected startup time: 30-60 seconds**

### Step 4: Verify Services
```powershell
# Check service status
docker-compose ps

# All services should show "healthy" or "running"
```

### Step 5: Run Health Checks
```powershell
# Comprehensive health check
make health-check

# Expected output:
# ✅ PostgreSQL OK - PostgreSQL 15 on ...
# ✅ RabbitMQ OK - Connection successful
# ✅ Redis OK - PONG received
# ✅ MinIO OK - All 4 buckets present
# ✅ Prometheus OK - API responding
# ✅ Grafana OK - API responding
```

If all pass: **✅ Phase 1 Infrastructure is Deployed!**

---

## 📊 Detailed Deployment Steps

### 1. Environment Setup

**Check if .env exists:**
```powershell
Test-Path .\.env
```

**If not, create it:**
```powershell
Copy-Item .env.example .env
Write-Host "Created .env from template"
```

**Review configuration:**
```powershell
Get-Content .env | Select-String "POSTGRES|RABBITMQ|REDIS|MINIO"
```

### 2. Start Services

**Launch infrastructure:**
```powershell
docker-compose up -d
```

**Output should show:**
```
[+] Running 9/9
 ✔ Network heimdall-network  Created
 ✔ Container heimdall-postgres  Started
 ✔ Container heimdall-rabbitmq  Started
 ✔ Container heimdall-redis  Started
 ✔ Container heimdall-minio  Started
 ✔ Container heimdall-prometheus  Started
 ✔ Container heimdall-grafana  Started
 (and others)
```

### 3. Monitor Startup

**Watch initialization logs:**
```powershell
docker-compose logs -f
```

**Look for these messages:**
- `"PostgreSQL X.Y starting"` → PostgreSQL ready
- `"Init script executed successfully"` → Schema created
- `"Management UI available at"` → RabbitMQ ready
- `"Ready to accept connections"` → All services ready

**Press Ctrl+C to exit logs**

### 4. Verify Services

**Check all containers running:**
```powershell
docker-compose ps
```

**Expected output:**
```
NAME                    STATUS
heimdall-postgres       Up (healthy)
heimdall-pgadmin        Up
heimdall-rabbitmq       Up (healthy)
heimdall-redis          Up (healthy)
heimdall-redis-commander Up
heimdall-minio          Up (healthy)
heimdall-minio-init     Exited
heimdall-prometheus     Up (healthy)
heimdall-grafana        Up (healthy)
```

### 5. Health Verification

**Run comprehensive checks:**
```powershell
make health-check
```

**Or manually test each service:**

**PostgreSQL:**
```powershell
docker-compose exec postgres psql -U heimdall_user -d heimdall -c "SELECT version();"
```

**RabbitMQ:**
```powershell
docker-compose exec rabbitmq rabbitmq-diagnostics ping
```

**Redis:**
```powershell
docker-compose exec redis redis-cli -a changeme ping
```

**MinIO:**
```powershell
curl http://localhost:9000/minio/health/live
```

**Prometheus:**
```powershell
curl http://localhost:9090/-/healthy
```

**Grafana:**
```powershell
curl http://localhost:3000/api/health
```

---

## 🎯 Next Actions

### Access Dashboards

```powershell
# PostgreSQL Management
Start-Process http://localhost:5050  # pgAdmin

# Message Queue
Start-Process http://localhost:15672 # RabbitMQ (guest/guest)

# Redis Cache
Start-Process http://localhost:8081  # Redis Commander

# Object Storage
Start-Process http://localhost:9001  # MinIO (minioadmin/minioadmin)

# Monitoring
Start-Process http://localhost:9090  # Prometheus
Start-Process http://localhost:3000  # Grafana (admin/admin)
```

### Database Operations

**Connect to PostgreSQL:**
```powershell
make postgres-connect
```

**List tables:**
```sql
\dt heimdall.*
```

**Check schema:**
```sql
SELECT table_name FROM information_schema.tables WHERE table_schema = 'heimdall';
```

**Verify TimescaleDB:**
```sql
SELECT * FROM timescaledb_information.hypertables;
```

### View Service Logs

```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f postgres
docker-compose logs -f rabbitmq
docker-compose logs -f redis

# Recent logs only
docker-compose logs --tail=50 postgres
```

---

## ✅ Phase 1 Completion Checklist

After deployment, verify:

- [ ] All docker-compose services show "healthy" or "running"
- [ ] Database schema created with 8 tables
- [ ] MinIO has 4 buckets created
- [ ] RabbitMQ management UI accessible
- [ ] Redis responding to PING
- [ ] Prometheus scraping targets
- [ ] Grafana can connect to Prometheus
- [ ] `make health-check` passes with ✅
- [ ] Can access PostgreSQL CLI
- [ ] Can access all dashboards

---

## 🔧 Common Operations

### Stop Infrastructure
```powershell
docker-compose down
```

### Restart Services
```powershell
docker-compose restart

# Or specific service
docker-compose restart postgres
```

### Full Cleanup (⚠️ Deletes data!)
```powershell
docker-compose down -v --remove-orphans
```

### View Logs
```powershell
# Live logs from all services
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs postgres
```

---

## 🐛 Troubleshooting

### Docker Not Running
**Error**: "Cannot connect to Docker daemon"

**Solution**:
1. Start Docker Desktop manually
2. Wait for Docker icon in system tray
3. Run `docker ps` to verify
4. Retry `docker-compose up -d`

### Port Already in Use
**Error**: "Address already in use"

**Solution**:
1. Edit `.env` file to change port
2. Example: Change `POSTGRES_PORT=5432` to `5433`
3. Retry `docker-compose up -d`

### Services Not Healthy
**Error**: Services stuck in "starting" state

**Solution**:
```powershell
# Check logs
docker-compose logs postgres

# Restart services
docker-compose restart

# Full cleanup and restart
docker-compose down -v
docker-compose up -d
```

### Health Check Failures
**Error**: "❌ FAILED - Connection refused"

**Solution**:
1. Wait 30-60 seconds for full startup
2. Check service logs: `docker-compose logs <service>`
3. Verify ports are open: `netstat -an | findstr 5432`
4. Restart failing service: `docker-compose restart postgres`

### Out of Memory
**Error**: "Cannot allocate memory"

**Solution**:
1. Close other applications
2. Increase Docker Desktop memory:
   - Docker Desktop → Settings → Resources
   - Set Memory to 6-8GB
   - Set CPUs to 4+
3. Restart Docker
4. Retry deployment

### No Disk Space
**Error**: "No space left on device"

**Solution**:
```powershell
# Clean up old Docker resources
docker system prune -a

# Check disk space
Get-Volume | Select-Object DriveLetter, SizeRemaining

# Free up space, then retry
```

---

## 📚 Documentation Links

| Document                                   | Purpose              |
| ------------------------------------------ | -------------------- |
| [PHASE1_GUIDE.md](PHASE1_GUIDE.md)         | Complete setup guide |
| [PHASE1_CHECKLIST.md](PHASE1_CHECKLIST.md) | Task tracking        |
| [PHASE1_STATUS.md](PHASE1_STATUS.md)       | Project status       |
| [README.md](README.md)                     | Project overview     |
| [AGENTS.md](AGENTS.md)                     | Phase roadmap        |

---

## ✨ What Happens Next?

### Phase 1 Complete ✅
- Infrastructure services running
- Database schema initialized
- Monitoring stack operational
- Health checks passing

### Phase 2: Core Services
- FastAPI service templates
- Celery integration
- Service scaffolding
- Logging configuration

### Phase 3+: Additional Phases
- Signal processing
- ML pipeline
- Frontend development
- Kubernetes deployment

---

## 📞 Support

**If deployment fails:**

1. Check prerequisites are met
2. Review logs: `docker-compose logs -f`
3. Check specific service: `docker-compose logs <service>`
4. Run health check: `make health-check`
5. Try full cleanup: `docker-compose down -v && docker-compose up -d`

**For detailed troubleshooting**, see [PHASE1_GUIDE.md](PHASE1_GUIDE.md#troubleshooting)

---

## 🎊 Success!

When you see all services healthy:

```
✅ PostgreSQL OK
✅ RabbitMQ OK
✅ Redis OK
✅ MinIO OK
✅ Prometheus OK
✅ Grafana OK

✅ All services healthy!
```

**Congratulations! Phase 1 Infrastructure is Deployed! 🎉**

Next: Start Phase 2 development

---

*Deployment Guide Created: 2025-10-22*  
*Status: Ready for deployment*  
*Expected Duration: 5 minutes setup + 60 seconds startup*

# Heimdall - Radio Source Localization

> *An intelligent platform for real-time radio signal localization*

![heimdall.png](heimdall.png)

[![License: CC Non-Commercial](https://img.shields.io/badge/License-CC%20Non--Commercial-orange.svg)](LICENSE)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-yellow.svg)](AGENTS.md)
[![Community: Amateur Radio](https://img.shields.io/badge/Community-Amateur%20Radio-blue.svg)](https://www.iaru.org/)

An AI-powered platform that locates radio transmissions in real-time using machine learning and distributed WebSDR receivers.

## Overview

Heimdall analyzes radio signals from multiple WebSDR stations to triangulate transmission sources. The system uses neural networks trained on radio propagation data to predict location coordinates with uncertainty estimates.

**Key specifications:**
- Target accuracy: ±30m (68% confidence)
- Processing latency: <500ms
- Network: 7 distributed WebSDR receivers
- Frequency bands: 2m/70cm amateur radio

## Architecture

- **Backend**: Python microservices (FastAPI, Celery)
- **ML Pipeline**: PyTorch Lightning with MLflow tracking
- **Frontend**: React + TypeScript + Mapbox
- **Infrastructure**: PostgreSQL + TimescaleDB, Redis, RabbitMQ, MinIO
- **Deployment**: Kubernetes with Helm charts

## Applications

**Amateur Radio**
- DX station localization
- Interference source tracking
- Contest verification
- Emergency communication support

**Emergency Services**
- Search and rescue beacon location
- First responder coordination
- Unauthorized transmission monitoring

**Research**
- Radio propagation studies
- Spectrum management
- Educational demonstrations

## Technical Details

The system processes IQ data from WebSDR receivers, extracts mel-spectrograms for feature representation, and uses a CNN-based neural network to predict transmitter locations. A Gaussian negative log-likelihood loss function enables uncertainty quantification for each prediction.

### Performance Characteristics (Phase 4 Validated)

**API Performance**
- Task submission latency: **~52ms average** (well under 100ms SLA)
- P95 latency: **52.81ms** (consistent performance)
- P99 latency: **62.63ms** (stable under load)
- Success rate: **100%** on 50 concurrent submissions

**System Processing**
- RF Acquisition per WebSDR: **63-70 seconds** (network-bound, expected)
- Database operations: **<50ms** per measurement insertion
- Message queue latency: **<100ms** for task routing
- Container memory footprint: **100-300MB** per service (efficient)

**Infrastructure Throughput**
- Concurrent task handling: **50+ simultaneous RF acquisitions** verified
- RabbitMQ routing: **reliable under production load**
- Redis caching: **<50ms per operation**
- TimescaleDB: **stable high-velocity ingestion**

## Development Status

**Phase 6: Inference Service** ✅ COMPLETE

- ✅ Phase 0: Repository Setup (Complete)
- ✅ Phase 1: Infrastructure & Database (Complete)
- ✅ Phase 2: Core Services Scaffolding (Complete)
- ✅ Phase 3: RF Acquisition Service (Complete)
- ✅ Phase 4: Data Ingestion & Validation (Complete - Infrastructure Verified)
  - E2E tests: 7/8 passing (87.5%)
  - Docker infrastructure: 13/13 containers healthy
  - Performance benchmarking: All SLAs met
  - Load testing: 50 concurrent tasks, 100% success rate
  - [Full Phase 4 Report →](docs/agents/20251022_080000_phase4_completion_final.md)

**Phase 5: Training Pipeline** ✅ COMPLETE
- ML pipeline development with PyTorch Lightning
- Model training with MLflow tracking
- [Phase 5 Handoff →](docs/agents/20251022_080000_phase5_handoff.md)

**Phase 6: Inference Service** ✅ COMPLETE
- Real-time inference with ONNX runtime
- Redis caching for optimized performance
- [Phase 6 Start Guide →](docs/agents/20251023_153000_phase6_start_here.md)

### Quick Start

```bash
# Clone repository
git clone https://github.com/fulgidus/heimdall.git
cd heimdall

# Setup environment (copy .env template)
cp .env.example .env
# Edit .env with your configuration

# Start infrastructure (requires Docker)
docker-compose up -d

# Verify services are healthy
make health-check
```

See [Phase 1 Guide](docs/agents/20251022_080000_phase1_guide.md) for detailed setup instructions.

### Development Credentials

For local development, all services use default credentials documented in [Development Default Credentials Guide](docs/dev-credentials.md).

**⚠️ Important:** These are development-only credentials. See the guide for:
- Default usernames and passwords for all services
- Connection strings and login examples
- How to access web UIs (Grafana, pgAdmin, RabbitMQ, MinIO, etc.)
- Security best practices and password rotation

**Quick reference:**
- PostgreSQL: `heimdall_user` / `changeme` @ `localhost:5432`
- Grafana: `admin` / `admin` @ http://localhost:3000
- RabbitMQ: `guest` / `guest` @ http://localhost:15672
- MinIO: `minioadmin` / `minioadmin` @ http://localhost:9001

📖 **Full credentials documentation:** [docs/dev-credentials.md](docs/dev-credentials.md)

## License

Creative Commons Non-Commercial. Developed by fulgidus for the amateur radio community.

## 🎯 Mission Statement

**Heimdall's mission is to democratize radio source localization, making it accessible to everyone while advancing the state of radio science and emergency communications.**

We believe that **radio waves belong to everyone**, and everyone should have the tools to understand and explore them. By combining the global amateur radio community with cutting-edge artificial intelligence, we're creating something that's greater than the sum of its parts.

---

## 🌟 The Team

**Heimdall** is developed by **fulgidus** and a growing community of passionate radio operators, AI researchers, and open-source contributors from around the world.

---

## 🚀 Ready to See the Invisible?

**The radio spectrum has been hidden in plain sight for over a century.**  
**Today, we make it visible.**  
**Tomorrow, we make it yours.**

### [🌟 Start Your Journey →](https://fulgidus.github.io/heimdall)

---

*Heimdall - Where Radio Waves Meet Artificial Intelligence*

---
---

# README - Italiano

> *Una piattaforma intelligente per la localizzazione in tempo reale di sorgenti radio*

![heimdall.png](heimdall.png)

[![License: CC Non-Commercial](https://img.shields.io/badge/License-CC%20Non--Commercial-orange.svg)](LICENSE)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Sviluppo-yellow.svg)](AGENTS.md)
[![Community: Amateur Radio](https://img.shields.io/badge/Community-Amateur%20Radio-blue.svg)](https://www.iaru.org/)

Una piattaforma basata su intelligenza artificiale che localizza trasmissioni radio in tempo reale utilizzando machine learning e ricevitori WebSDR distribuiti.

## Panoramica

Heimdall analizza segnali radio da multiple stazioni WebSDR per triangolare le sorgenti di trasmissione. Il sistema utilizza reti neurali addestrate su dati di propagazione radio per predire coordinate di posizione con stime di incertezza.

**Specifiche chiave:**
- Precisione target: ±30m (68% di confidenza)
- Latenza di elaborazione: <500ms
- Rete: 7 ricevitori WebSDR distribuiti
- Bande di frequenza: radioamatori 2m/70cm

## Architettura

- **Backend**: Microservizi Python (FastAPI, Celery)
- **Pipeline ML**: PyTorch Lightning con tracking MLflow
- **Frontend**: React + TypeScript + Mapbox
- **Infrastruttura**: PostgreSQL + TimescaleDB, Redis, RabbitMQ, MinIO
- **Deployment**: Kubernetes con Helm charts

## Applicazioni

**Radioamatori**
- Localizzazione stazioni DX
- Tracciamento sorgenti di interferenza
- Verifica contest
- Supporto comunicazioni di emergenza

**Servizi di Emergenza**
- Localizzazione beacon di ricerca e soccorso
- Coordinamento primi soccorritori
- Monitoraggio trasmissioni non autorizzate

**Ricerca**
- Studi di propagazione radio
- Gestione dello spettro
- Dimostrazioni educative

## Dettagli Tecnici

Il sistema elabora dati IQ dai ricevitori WebSDR, estrae mel-spettrogrammi per la rappresentazione delle caratteristiche, e utilizza una rete neurale basata su CNN per predire le posizioni dei trasmettitori. Una funzione di perdita Gaussiana a log-verosimiglianza negativa abilita la quantificazione dell'incertezza per ogni predizione.

### Caratteristiche di Performance (Fase 4 Validata)

**Performance API**
- Latenza sottomissione task: **~52ms media** (ben sotto SLA di 100ms)
- Latenza P95: **52.81ms** (performance consistente)
- Latenza P99: **62.63ms** (stabile sotto carico)
- Tasso di successo: **100%** su 50 sottomissioni concorrenti

**Elaborazione Sistema**
- Acquisizione RF per WebSDR: **63-70 secondi** (limitato dalla rete, previsto)
- Operazioni database: **<50ms** per inserimento misura
- Latenza coda messaggi: **<100ms** per routing task
- Footprint memoria container: **100-300MB** per servizio (efficiente)

**Throughput Infrastruttura**
- Gestione task concorrenti: **50+ acquisizioni RF simultanee** verificate
- Routing RabbitMQ: **affidabile sotto carico produzione**
- Caching Redis: **<50ms per operazione**
- TimescaleDB: **ingestion ad alta velocità stabile**

## Stato di Sviluppo

**Fase 6: Servizio Inferenza** ✅ COMPLETA

- ✅ Fase 0: Setup Repository (Completa)
- ✅ Fase 1: Infrastruttura & Database (Completa)
- ✅ Fase 2: Scaffolding Servizi Core (Completa)
- ✅ Fase 3: Servizio Acquisizione RF (Completa)
- ✅ Fase 4: Data Ingestion & Validazione (Completa - Infrastruttura Verificata)
  - Test E2E: 7/8 passati (87.5%)
  - Infrastruttura Docker: 13/13 container sani
  - Benchmarking performance: Tutti gli SLA rispettati
  - Load testing: 50 task concorrenti, 100% tasso di successo
  - [Report completo Fase 4 →](docs/agents/20251022_080000_phase4_completion_final.md)

**Fase 5: Training Pipeline** ✅ COMPLETA
- Sviluppo pipeline ML con PyTorch Lightning
- Training modello con tracking MLflow
- [Handoff Fase 5 →](docs/agents/20251022_080000_phase5_handoff.md)

**Fase 6: Servizio Inferenza** ✅ COMPLETA
- Inferenza in tempo reale con ONNX runtime
- Caching Redis per performance ottimizzate
- [Guida Avvio Fase 6 →](docs/agents/20251023_153000_phase6_start_here.md)

### Avvio Rapido

```bash
# Clona repository
git clone https://github.com/fulgidus/heimdall.git
cd heimdall

# Setup environment (copia template .env)
cp .env.example .env
# Modifica .env con la tua configurazione

# Avvia infrastruttura (richiede Docker)
docker-compose up -d

# Verifica che i servizi siano attivi
make health-check
```

Vedi [Guida Fase 1](docs/agents/20251022_080000_phase1_guide.md) per istruzioni di setup dettagliate.

### Credenziali di Sviluppo

Per lo sviluppo locale, tutti i servizi utilizzano credenziali predefinite documentate nella [Guida Credenziali Predefinite di Sviluppo](docs/dev-credentials.md).

**⚠️ Importante:** Queste sono credenziali solo per sviluppo. Consulta la guida per:
- Username e password predefinite per tutti i servizi
- Stringhe di connessione ed esempi di login
- Come accedere alle interfacce web (Grafana, pgAdmin, RabbitMQ, MinIO, ecc.)
- Best practice di sicurezza e rotazione password

**Riferimento rapido:**
- PostgreSQL: `heimdall_user` / `changeme` @ `localhost:5432`
- Grafana: `admin` / `admin` @ http://localhost:3000
- RabbitMQ: `guest` / `guest` @ http://localhost:15672
- MinIO: `minioadmin` / `minioadmin` @ http://localhost:9001

📖 **Documentazione completa credenziali:** [docs/dev-credentials.md](docs/dev-credentials.md)

## Licenza

Creative Commons Non-Commercial. Sviluppato da fulgidus per la comunità radioamatoriale.

## 🎯 Mission Statement

**La missione di Heimdall è democratizzare la localizzazione di sorgenti radio, rendendola accessibile a tutti mentre si avanza lo stato dell'arte della scienza radio e delle comunicazioni di emergenza.**

Crediamo che **le onde radio appartengano a tutti**, e tutti dovrebbero avere gli strumenti per comprenderle ed esplorarle. Combinando la comunità radioamatoriale globale con l'intelligenza artificiale all'avanguardia, stiamo creando qualcosa che è più grande della somma delle sue parti.

---

## 🌟 Il Team

**Heimdall** è sviluppato da **fulgidus** e una crescente comunità di appassionati operatori radio, ricercatori AI, e contributori open-source da tutto il mondo.

---

## 🚀 Pronti a Vedere l'Invisibile?

**Lo spettro radio è stato nascosto in bella vista per oltre un secolo.**  
**Oggi, lo rendiamo visibile.**  
**Domani, lo rendiamo vostro.**

### [🌟 Inizia il Tuo Viaggio →](https://fulgidus.github.io/heimdall)

---

*Heimdall - Dove le Onde Radio Incontrano l'Intelligenza Artificiale*
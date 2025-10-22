# Heimdall SDR - Quick Setup Guide (Windows PowerShell)

Questa guida ti aiuta a mettere in piedi l'ambiente di sviluppo Heimdall su Windows con PowerShell.

## Prerequisiti

- **Python 3.11+** installato
- **Docker Desktop** installato (scarica da https://www.docker.com/products/docker-desktop)
- **Git** (per clonare il repository)
- **Node.js 18+** (per il frontend, facoltativo per ora)

## Quick Setup (5 minuti)

### 1. Clona il repository e accedi alla cartella

```powershell
git clone https://github.com/fulgidus/heimdall.git
cd heimdall
```

### 2. Crea e attiva virtualenv Python

```powershell
# Crea virtualenv
python -m venv .venv

# Attiva virtualenv
.venv\Scripts\Activate.ps1

# Se ricevi errore di esecuzione, esegui:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Verifica:** dovresti vedere `(.venv)` all'inizio della riga di prompt PowerShell.

### 3. Esegui lo script di setup automatico

```powershell
.\dev-setup.ps1
```

Questo script farà:
- ✅ Verifica Python, pip e virtualenv
- ✅ Crea/copia il file `.env` (variabili d'ambiente)
- ✅ Verifica Docker Desktop
- ✅ Installa dipendenze Python di tutti i microservizi
- ✅ Ti mostra i prossimi passi

### 4. Avvia Docker Desktop

Apri il menu Start di Windows e cerca **"Docker Desktop"**. Clicca per avviarlo. Aspetta che appaia l'icona di Docker nella barra di sistema (in basso a destra).

### 5. Avvia l'infrastruttura Docker (nel terminale PowerShell)

```powershell
# Assicurati di essere nella cartella root del progetto
docker compose up -d

# Verifica che i servizi siano attivi
docker compose ps
```

Dovresti vedere container di PostgreSQL, Redis, RabbitMQ, MinIO, ecc. in stato "running" (o "healthy").

### 6. Avvia i microservizi Python in nuove finestre PowerShell

```powershell
.\scripts\start-microservices.ps1
```

Questo apre 5 nuove finestre PowerShell, una per servizio:
- API Gateway (port 8000)
- RF Acquisition (port 8001)
- Training (port 8002)
- Inference (port 8003)
- Data Ingestion Web (port 8004)

Aspetta ~10 secondi che si avviino. Dovresti vedere messaggi di Uvicorn in ciascuna finestra.

### 7. Accedi ai servizi

Apri il browser e visita:

- **API Gateway Docs:** http://localhost:8000/docs
- **RF Acquisition Docs:** http://localhost:8001/docs
- **Training Docs:** http://localhost:8002/docs
- **Inference Docs:** http://localhost:8003/docs
- **Data Ingestion Web Docs:** http://localhost:8004/docs
- **Postgres Admin (pgAdmin):** http://localhost:5050 (admin@pg.com / admin)
- **RabbitMQ Management:** http://localhost:15672 (guest / guest)
- **MinIO Console:** http://localhost:9001 (minioadmin / minioadmin)

## Comandi Essenziali

### Gestire l'infrastruttura Docker

```powershell
# Avvia infrastruttura
docker compose up -d

# Ferma infrastruttura
docker compose down

# Visualizza log di tutti i servizi
docker compose logs -f

# Log di un singolo servizio
docker compose logs -f postgres      # o redis, rabbitmq, minio, ecc.

# Riavvia un servizio
docker compose restart postgres

# Ricrea i container (cancella dati!)
docker compose down -v && docker compose up -d
```

### Gestire i microservizi Python

```powershell
# Esegui test di import
.\scripts\test-all-services.ps1

# Esegui test pytest (con virtualenv attivo)
pytest -q

# Test di un servizio specifico
pytest services/rf-acquisition/tests -q

# Esegui health-check API Gateway
curl http://localhost:8000/health
# o con PowerShell:
Invoke-RestMethod -Uri http://localhost:8000/health -Method Get

# Health-check microservizi
.\scripts\health-check-microservices.ps1
```

### Database

```powershell
# Connetti a PostgreSQL
docker compose exec postgres psql -U heimdall_user -d heimdall

# Visualizza tabelle
\dt

# Esci dalla console
\q
```

### Redis

```powershell
# Connetti a Redis
docker compose exec redis redis-cli

# Visualizza chiavi
KEYS *

# Esci
exit
```

## Disattivare virtualenv (quando finisci)

```powershell
deactivate
```

## Troubleshooting

### Errore: "Docker daemon non responde"

- Assicurati che **Docker Desktop sia in esecuzione** (menu Start > Docker Desktop)
- Attendi 30-60 secondi dopo il primo avvio
- Se continua, riavvia Docker: menu Start > Cerca "Services" > trova "Docker Desktop" > riavvia

### Errore: "port 8000 already in use"

Un altro processo sta usando la porta. Opzioni:
1. Ferma il processo che occupa la porta
2. Cambia la porta nel file `.env` (es. `API_GATEWAY_PORT=8001`)
3. Riavvia il computer

### Errore: "psycopg2 module not found"

```powershell
# Assicurati virtualenv attivo, poi:
pip install psycopg2-binary
```

### Virtualenv non attivo dopo riavvio shell

Ricorda di attivare virtualenv in ogni nuova finestra PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

### "Cannot find path .venv\Scripts\Activate.ps1"

Se sei su Windows con PowerShell e l'errore è di tipo "cannot find", prova:
```powershell
# Dai permessi di esecuzione agli script PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Prova di nuovo
.venv\Scripts\Activate.ps1
```

## Prossimi passi

- Leggi [SETUP.md](SETUP.md) per dettagli avanzati
- Guarda [AGENTS.md](AGENTS.md) per la roadmap di sviluppo
- Vedi [PHASE3_TO_PHASE4_HANDOFF.md](PHASE3_TO_PHASE4_HANDOFF.md) per lo stato attuale del progetto

## Supporto

Se hai domande o problemi:
1. Controlla i log: `docker compose logs -f` o apri le finestre PowerShell dei servizi
2. Verifica che virtualenv sia attivo
3. Assicurati che Docker Desktop sia in esecuzione
4. Leggi [SETUP.md](SETUP.md) per guide dettagliate

---

**Sei pronto? Inizia con il Passo 1 sopra! 🚀**

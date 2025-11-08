#!/usr/bin/env python
"""
Entrypoint che avvia sia l'API FastAPI che il Celery worker.
Usa subprocess per avviare entrambi in parallelo dentro lo stesso container.
"""
import os
import signal
import subprocess
import sys


# Handlers per shutdown graceful
def signal_handler(sig, frame):
    print("Shutting down...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def start_api():
    """Avvia l'API Uvicorn."""
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8001",
        "--log-level",
        log_level,
    ]
    print(f"[API] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def start_worker():
    """Avvia il Celery worker."""
    # Aspetta un po' prima di avviare il worker (per permettere all'API di partire)
    import time

    time.sleep(2)

    log_level = os.getenv("LOG_LEVEL", "info").upper()
    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "src.main:celery_app",  # Correct path to celery_app in main.py
        "worker",
        "--loglevel",
        log_level,
        "--concurrency",
        "4",
        "--prefetch-multiplier",
        "1",
        "--time-limit",
        "1800",
        "--soft-time-limit",
        "1500",
        "--queues",
        "celery",  # Listen only on default celery queue (not training)
    ]
    print(f"[WORKER] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def start_beat():
    """Avvia il Celery Beat scheduler per periodic tasks."""
    # Aspetta un po' prima di avviare il beat (per permettere all'API di partire)
    import time

    time.sleep(2)

    # Rimuovi il pidfile se esiste già (cleanup da restart precedenti)
    pidfile = "/tmp/celerybeat.pid"
    if os.path.exists(pidfile):
        print(f"[BEAT] Removing stale pidfile: {pidfile}")
        os.remove(pidfile)

    log_level = os.getenv("LOG_LEVEL", "info").upper()
    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "src.main:celery_app",  # Correct path to celery_app in main.py
        "beat",
        "--loglevel",
        log_level,
        "--scheduler",
        "celery.beat:PersistentScheduler",
        "--pidfile",
        pidfile,
        "--schedule",
        "/tmp/celerybeat-schedule",
    ]
    print(f"[BEAT] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


if __name__ == "__main__":
    print("RF Acquisition Service - Dual-mode Entrypoint")
    print("=" * 50)

    # Avvia API, Worker E Beat
    api_process = start_api()
    worker_process = start_worker()
    beat_process = start_beat()

    print("[MAIN] All processes started")
    print(f"[MAIN] API PID: {api_process.pid}")
    print(f"[MAIN] Worker PID: {worker_process.pid}")
    print(f"[MAIN] Beat PID: {beat_process.pid}")

    try:
        # Aspetta tutti e tre i processi
        while True:
            # Check se i processi sono ancora vivi
            api_retcode = api_process.poll()
            worker_retcode = worker_process.poll()
            beat_retcode = beat_process.poll()

            if api_retcode is not None:
                print(f"[ERROR] API process exited with code {api_retcode}")
                worker_process.terminate()
                beat_process.terminate()
                sys.exit(api_retcode)

            if worker_retcode is not None:
                print(f"[ERROR] Worker process exited with code {worker_retcode}")
                api_process.terminate()
                beat_process.terminate()
                sys.exit(worker_retcode)

            if beat_retcode is not None:
                print(f"[ERROR] Beat process exited with code {beat_retcode}")
                api_process.terminate()
                worker_process.terminate()
                sys.exit(beat_retcode)

            # Sleep a bit to avoid busy waiting
            import time

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[MAIN] Received interrupt, terminating processes...")
        api_process.terminate()
        worker_process.terminate()
        beat_process.terminate()
        api_process.wait(timeout=10)
        worker_process.wait(timeout=10)
        beat_process.wait(timeout=10)
        sys.exit(0)

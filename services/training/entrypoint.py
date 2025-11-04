#!/usr/bin/env python
"""
Entrypoint that starts both the FastAPI API and the Celery worker.
Uses subprocess to start both in parallel inside the same container.
"""
import os
import signal
import subprocess
import sys


# Handlers for graceful shutdown
def signal_handler(sig, frame):
    print("Shutting down...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def start_api():
    """Start the Uvicorn API server."""
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8002",
        "--log-level",
        log_level,
    ]
    print(f"[API] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def start_worker():
    """Start the Celery worker."""
    # Wait a bit before starting the worker (to allow API to start)
    import time

    time.sleep(2)

    log_level = os.getenv("LOG_LEVEL", "info").upper()
    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "src.main:celery_app",
        "worker",
        "--loglevel",
        log_level,
        "--pool",
        "solo",  # Use solo pool to avoid daemon process issues with DataLoader workers
        "--time-limit",
        "21600",  # 6 hours
        "--soft-time-limit",
        "19800",  # 5.5 hours
        "-Q",
        "training",  # Listen only on training queue
    ]
    print(f"[WORKER] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


if __name__ == "__main__":
    print("Training Service - Dual-mode Entrypoint")
    print("=" * 50)

    # Start API and Worker
    api_process = start_api()
    worker_process = start_worker()

    print("[MAIN] All processes started")
    print(f"[MAIN] API PID: {api_process.pid}")
    print(f"[MAIN] Worker PID: {worker_process.pid}")

    try:
        # Wait for both processes
        while True:
            # Check if processes are still alive
            api_retcode = api_process.poll()
            worker_retcode = worker_process.poll()

            if api_retcode is not None:
                print(f"[ERROR] API process exited with code {api_retcode}")
                worker_process.terminate()
                sys.exit(api_retcode)

            if worker_retcode is not None:
                print(f"[ERROR] Worker process exited with code {worker_retcode}")
                api_process.terminate()
                sys.exit(worker_retcode)

            # Sleep a bit to avoid busy waiting
            import time

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[MAIN] Received interrupt, terminating processes...")
        api_process.terminate()
        worker_process.terminate()
        api_process.wait(timeout=10)
        worker_process.wait(timeout=10)
        sys.exit(0)

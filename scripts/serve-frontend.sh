#!/bin/bash
set -e

ENV=${1:-development}
PORT=${2:-5173}

echo "Starting frontend dev server: $ENV:$PORT"

cd "$(dirname "$0")/../frontend"

# Load environment
if [ -f ".env.$ENV" ]; then
  export $(cat .env.$ENV | grep -v '^#' | xargs)
fi

npm run dev -- --port $PORT

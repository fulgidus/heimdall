#!/bin/bash
set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
ENV=${1:-development}
VERBOSE=${2:-false}

echo -e "${YELLOW}Building frontend for environment: $ENV${NC}"

# Validate environment
if [[ ! "$ENV" =~ ^(development|staging|production)$ ]]; then
  echo -e "${RED}Invalid environment: $ENV${NC}"
  echo "Usage: $0 [development|staging|production] [verbose]"
  exit 1
fi

# Setup environment
cd "$(dirname "$0")/../frontend"

# Load environment variables
if [ -f ".env.$ENV" ]; then
  echo -e "${GREEN}Loading environment: .env.$ENV${NC}"
  export $(cat .env.$ENV | grep -v '^#' | xargs)
fi

# Clean previous build
echo -e "${YELLOW}Cleaning previous build...${NC}"
rm -rf dist/ node_modules/.vite

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
npm ci

# Build
echo -e "${YELLOW}Building...${NC}"
npm run build -- --mode $ENV

# Generate manifest
echo -e "${GREEN}Generating build manifest...${NC}"
cat > dist/manifest.json <<EOF
{
  "version": "$(git describe --tags --always)",
  "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
  "environment": "$ENV",
  "commit": "$(git rev-parse HEAD)",
  "author": "$(git config user.name)"
}
EOF

# Report
echo -e "${GREEN}Build complete!${NC}"
echo "Output: dist/"
du -sh dist/
find dist/ -type f -name "*.js" -o -name "*.css" | wc -l
echo -e "${GREEN}files generated${NC}"

#!/bin/bash
# Get Keycloak access token for testing
# Reads credentials from .env file

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(cat .env | sed 's/\r$//' | grep -v '^#' | grep -v '^$' | xargs)
elif [ -f .env.example ]; then
    export $(cat .env.example | sed 's/\r$//' | grep -v '^#' | grep -v '^$' | xargs)
else
    echo "ERROR: No .env or .env.example file found"
    exit 1
fi

# Use environment variables or defaults
KEYCLOAK_URL="${KEYCLOAK_URL:-http://localhost:8080}"
KEYCLOAK_REALM="${KEYCLOAK_REALM:-heimdall}"
APP_USER_EMAIL="${APP_USER_EMAIL:-admin@heimdall.local}"
APP_USER_PASSWORD="${APP_USER_PASSWORD:-admin}"
CLIENT_ID="${KEYCLOAK_FRONTEND_CLIENT_ID:-heimdall-frontend}"

# Get token from Keycloak
TOKEN_RESPONSE=$(curl -s -X POST \
  "${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=${CLIENT_ID}" \
  -d "username=${APP_USER_EMAIL}" \
  -d "password=${APP_USER_PASSWORD}")

# Extract access token
ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

if [ -z "$ACCESS_TOKEN" ] || [ "$ACCESS_TOKEN" = "null" ]; then
    echo "ERROR: Failed to get access token from Keycloak" >&2
    echo "Response: $TOKEN_RESPONSE" >&2
    exit 1
fi

echo "$ACCESS_TOKEN"

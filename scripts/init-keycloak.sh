#!/bin/sh

# Keycloak initialization script - creates realm, client, and users

set -e

KC_SERVER="http://keycloak:8080"
KC_REALM="heimdall"
KC_CLIENT_ID="heimdall-frontend"
KC_CLIENT_SECRET="${KEYCLOAK_CLIENT_SECRET:-heimdall-secret}"
ADMIN_USER="${KEYCLOAK_ADMIN:-admin}"
ADMIN_PASS="${KEYCLOAK_ADMIN_PASSWORD:-admin}"

# Wait for Keycloak to be ready
echo "Waiting for Keycloak to be ready..."
for i in {1..30}; do
  if curl -s "$KC_SERVER/health/ready" > /dev/null 2>&1; then
    echo "Keycloak is ready!"
    break
  fi
  echo "Attempt $i: Waiting for Keycloak..."
  sleep 2
done

# Get admin token
echo "Getting admin token..."
TOKEN=$(curl -s -X POST \
  "$KC_SERVER/realms/master/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=admin-cli&username=$ADMIN_USER&password=$ADMIN_PASS&grant_type=password" \
  | jq -r '.access_token' 2>/dev/null || echo "")

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
  echo "Failed to get admin token. Keycloak might not be ready."
  exit 1
fi

# Check if realm exists
echo "Checking if realm '$KC_REALM' exists..."
REALM_EXISTS=$(curl -s "$KC_SERVER/admin/realms/$KC_REALM" \
  -H "Authorization: Bearer $TOKEN" \
  2>/dev/null | jq '.realm' 2>/dev/null || echo "")

if [ -z "$REALM_EXISTS" ] || [ "$REALM_EXISTS" = "null" ]; then
  echo "Creating realm '$KC_REALM'..."
  curl -s -X POST "$KC_SERVER/admin/realms" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "realm": "'$KC_REALM'",
      "enabled": true,
      "displayName": "Heimdall SDR",
      "accessTokenLifespan": 3600,
      "refreshTokenMaxReuse": 0,
      "refreshTokenLifespan": 86400,
      "accessCodeLifespan": 60,
      "accessCodeLifespanUserAction": 300,
      "actionTokenGeneratedByUserLifespan": 3600,
      "actionTokenGeneratedByAdminLifespan": 43200,
      "offlineSessionIdleTimeout": 2592000,
      "offlineSessionMaxLifespan": 5184000
    }' > /dev/null

  echo "Realm '$KC_REALM' created."
else
  echo "Realm '$KC_REALM' already exists."
fi

# Check if client exists
echo "Checking if client '$KC_CLIENT_ID' exists..."
CLIENT_ID=$(curl -s "$KC_SERVER/admin/realms/$KC_REALM/clients?clientId=$KC_CLIENT_ID" \
  -H "Authorization: Bearer $TOKEN" \
  2>/dev/null | jq -r '.[0].id // empty' 2>/dev/null || echo "")

if [ -z "$CLIENT_ID" ]; then
  echo "Creating client '$KC_CLIENT_ID'..."
  curl -s -X POST "$KC_SERVER/admin/realms/$KC_REALM/clients" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "clientId": "'$KC_CLIENT_ID'",
      "name": "Heimdall Frontend",
      "enabled": true,
      "public": false,
      "clientSecret": "'$KC_CLIENT_SECRET'",
      "redirectUris": [
        "http://localhost:3000/*",
        "http://localhost:3000/callback"
      ],
      "webOrigins": [
        "http://localhost:3000",
        "http://localhost:8000"
      ],
      "accessType": "confidential",
      "standardFlowEnabled": true,
      "implicitFlowEnabled": false,
      "directAccessGrantsEnabled": true,
      "serviceAccountsEnabled": false
    }' > /dev/null

  echo "Client '$KC_CLIENT_ID' created."
else
  echo "Client '$KC_CLIENT_ID' already exists with ID: $CLIENT_ID"
fi

# Check if user exists
echo "Checking if user 'admin' exists..."
USER_ID=$(curl -s "$KC_SERVER/admin/realms/$KC_REALM/users?username=admin" \
  -H "Authorization: Bearer $TOKEN" \
  2>/dev/null | jq -r '.[0].id // empty' 2>/dev/null || echo "")

if [ -z "$USER_ID" ]; then
  echo "Creating user 'admin'..."
  curl -s -X POST "$KC_SERVER/admin/realms/$KC_REALM/users" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "username": "admin",
      "email": "admin@heimdall.local",
      "firstName": "Admin",
      "lastName": "User",
      "enabled": true,
      "emailVerified": true,
      "credentials": [{
        "type": "password",
        "value": "admin",
        "temporary": false
      }]
    }' > /dev/null

  echo "User 'admin' created."
else
  echo "User 'admin' already exists with ID: $USER_ID"
fi

echo "Keycloak initialization complete!"

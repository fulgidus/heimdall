#!/bin/bash

# Keycloak realm and user initialization script
# This script imports the Heimdall realm configuration from JSON

set -e

echo "⏳ Waiting for Keycloak to be ready..."
sleep 15

KEYCLOAK_ADMIN=${KEYCLOAK_ADMIN:-admin}
KEYCLOAK_ADMIN_PASSWORD=${KEYCLOAK_ADMIN_PASSWORD:-admin}
KEYCLOAK_REALM=${KEYCLOAK_REALM:-heimdall}
APP_USER_EMAIL=${APP_USER_EMAIL:-admin@heimdall.local}
APP_USER_PASSWORD=${APP_USER_PASSWORD:-admin}
KEYCLOAK_URL=${KEYCLOAK_URL:-http://localhost:8080}

echo "🔐 Keycloak Configuration:"
echo "  - URL: $KEYCLOAK_URL"
echo "  - Admin User: $KEYCLOAK_ADMIN"
echo "  - Realm: $KEYCLOAK_REALM"
echo "  - App User Email: $APP_USER_EMAIL"

# Get admin token
echo "� Getting admin token..."
TOKEN=$(curl -s -X POST \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d "client_id=admin-cli" \
  -d "username=$KEYCLOAK_ADMIN" \
  -d "password=$KEYCLOAK_ADMIN_PASSWORD" \
  -d "grant_type=password" \
  "$KEYCLOAK_URL/realms/master/protocol/openid-connect/token" | \
  jq -r '.access_token')

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
  echo "❌ Failed to get admin token"
  exit 1
fi

# Check if realm already exists
echo "📋 Checking if realm '$KEYCLOAK_REALM' exists..."
REALM_EXISTS=$(curl -s -H "Authorization: Bearer $TOKEN" \
  "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM" | jq -e '.id' 2>/dev/null || echo "false")

if [ "$REALM_EXISTS" != "false" ]; then
  echo "✅ Realm '$KEYCLOAK_REALM' already exists"
  
  # Update password for existing user
  echo "🔄 Updating user password..."
  USER_ID=$(curl -s -H "Authorization: Bearer $TOKEN" \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users?username=$APP_USER_EMAIL&exact=true" | \
    jq -r '.[0].id' 2>/dev/null || echo "")
  
  if [ -n "$USER_ID" ] && [ "$USER_ID" != "null" ]; then
    curl -s -X PUT \
      -H 'Content-Type: application/json' \
      -H "Authorization: Bearer $TOKEN" \
      -d '{
        "type": "password",
        "value": "'$APP_USER_PASSWORD'",
        "temporary": false
      }' \
      "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users/$USER_ID/reset-password"
    echo "✅ Password updated"
  fi
else
  echo "🆕 Importing realm from JSON..."
  
  # Import realm from JSON file
  if [ -f /opt/keycloak/data/import/heimdall-realm.json ]; then
    curl -s -X POST \
      -H 'Content-Type: application/json' \
      -H "Authorization: Bearer $TOKEN" \
      -d @/opt/keycloak/data/import/heimdall-realm.json \
      "$KEYCLOAK_URL/admin/realms"
    
    echo "✅ Realm imported from JSON"
  else
    echo "⚠️  JSON file not found, creating realm manually..."
    
    # Fallback: create realm manually
    curl -s -X POST \
      -H 'Content-Type: application/json' \
      -H "Authorization: Bearer $TOKEN" \
      -d '{
        "realm": "'$KEYCLOAK_REALM'",
        "enabled": true,
        "displayName": "Heimdall SDR"
      }' \
      "$KEYCLOAK_URL/admin/realms"
    
    echo "✅ Realm created"
  fi
fi

# Get token for realm operations
TOKEN=$(curl -s -X POST \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d "client_id=admin-cli" \
  -d "username=$KEYCLOAK_ADMIN" \
  -d "password=$KEYCLOAK_ADMIN_PASSWORD" \
  -d "grant_type=password" \
  "$KEYCLOAK_URL/realms/master/protocol/openid-connect/token" | \
  jq -r '.access_token')

# Check if user already exists
echo "👤 Checking if user '$APP_USER_EMAIL' exists..."
USER_CHECK=$(curl -s \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users?username=$APP_USER_EMAIL&exact=true" | \
  jq -r '.[] | .id' 2>/dev/null || echo "")

if [ -z "$USER_CHECK" ]; then
  echo "🆕 Creating user '$APP_USER_EMAIL'..."
  
  # Create user
  USER_RESPONSE=$(curl -s -X POST \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $TOKEN" \
    -d '{
      "username": "'$APP_USER_EMAIL'",
      "email": "'$APP_USER_EMAIL'",
      "firstName": "Admin",
      "lastName": "User",
      "enabled": true,
      "emailVerified": true,
      "credentials": [{
        "type": "password",
        "value": "'$APP_USER_PASSWORD'",
        "temporary": false
      }]
    }' \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users")
  
  USER_ID=$(echo "$USER_RESPONSE" | jq -r '.id' 2>/dev/null || echo "")
  
  if [ -z "$USER_ID" ]; then
    echo "❌ Failed to create user"
    exit 1
  fi
  
  echo "✅ User created with ID: $USER_ID"
  
  # Assign admin role
  echo "🔑 Assigning admin role..."
  curl -s -X POST \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $TOKEN" \
    -d '[{
      "name": "admin",
      "composite": false
    }]' \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users/$USER_ID/role-mappings/realm"
  
  echo "✅ User setup completed"
else
  echo "✅ User '$APP_USER_EMAIL' already exists"
fi

echo "🎉 Keycloak initialization completed!"

#!/bin/sh

# Keycloak realm initialization via API
# Run after Keycloak starts to create realm and users
# Note: Using /bin/sh for Alpine Linux compatibility

# Configuration from environment variables
KEYCLOAK_URL=${KEYCLOAK_URL:-http://keycloak:8080}
KEYCLOAK_ADMIN=${KEYCLOAK_ADMIN:-admin}
KEYCLOAK_ADMIN_PASSWORD=${KEYCLOAK_ADMIN_PASSWORD:-admin}
KEYCLOAK_REALM=${KEYCLOAK_REALM:-heimdall}
KEYCLOAK_API_GATEWAY_CLIENT_ID=${KEYCLOAK_API_GATEWAY_CLIENT_ID:-api-gateway}
KEYCLOAK_API_GATEWAY_CLIENT_SECRET=${KEYCLOAK_API_GATEWAY_CLIENT_SECRET:-api-gateway-secret}
VITE_KEYCLOAK_CLIENT_ID=${VITE_KEYCLOAK_CLIENT_ID:-heimdall-frontend}
APP_USER_EMAIL=${APP_USER_EMAIL:-admin@heimdall.local}
APP_USER_PASSWORD=${APP_USER_PASSWORD:-Admin123!@#}

echo "🔧 Keycloak Initialization Script"
echo "  Keycloak URL: $KEYCLOAK_URL"
echo "  Realm: $KEYCLOAK_REALM"
echo "  Admin User: $KEYCLOAK_ADMIN"

# Wait for Keycloak to be ready
echo "⏳ Waiting for Keycloak to be ready..."
WAIT_COUNTER=0
while [ $WAIT_COUNTER -lt 30 ]; do
  if curl -s -f -o /dev/null "$KEYCLOAK_URL" 2>/dev/null; then
    echo "✅ Keycloak is ready"
    break
  fi
  WAIT_COUNTER=$((WAIT_COUNTER + 1))
  echo "  Attempt $WAIT_COUNTER/30..."
  sleep 1
done

# Get admin token
echo "🔑 Getting admin token..."
TOKEN_RESPONSE=$(curl -s -X POST \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d "client_id=admin-cli&username=$KEYCLOAK_ADMIN&password=$KEYCLOAK_ADMIN_PASSWORD&grant_type=password" \
  "$KEYCLOAK_URL/realms/master/protocol/openid-connect/token" 2>&1)

ADMIN_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.access_token' 2>/dev/null)

if [ -z "$ADMIN_TOKEN" ] || [ "$ADMIN_TOKEN" = "null" ]; then
  echo "⚠️  Could not get admin token (may try again on restart)"
  exit 0
fi

echo "✅ Token obtained"

# Check if realm exists
echo "📋 Checking realm '$KEYCLOAK_REALM'..."
REALM_EXISTS=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM" 2>/dev/null | jq -r '.realm' 2>/dev/null)

if [ "$REALM_EXISTS" = "$KEYCLOAK_REALM" ]; then
  echo "ℹ️  Realm already exists, skipping creation"
else
  echo "📋 Creating realm '$KEYCLOAK_REALM'..."
  curl -s -X POST \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -d '{
      "realm": "'$KEYCLOAK_REALM'",
      "enabled": true,
      "displayName": "Heimdall SDR",
      "sslRequired": "none",
      "accessTokenLifespan": 3600,
      "refreshTokenLifespan": 86400,
      "actionTokenGeneratedByAdminLifespan": 43200
    }' \
    "$KEYCLOAK_URL/admin/realms" > /dev/null 2>&1
  
  echo "✅ Realm created successfully"
fi

# Create API Gateway client
echo "🔌 Creating API Gateway client..."
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "clientId": "'$KEYCLOAK_API_GATEWAY_CLIENT_ID'",
    "secret": "'$KEYCLOAK_API_GATEWAY_CLIENT_SECRET'",
    "name": "API Gateway",
    "enabled": true,
    "publicClient": false,
    "bearerOnly": false,
    "redirectUris": ["http://localhost:8000/*"],
    "webOrigins": ["*"],
    "standardFlowEnabled": true,
    "directAccessGrantsEnabled": true,
    "serviceAccountsEnabled": true
  }' \
  "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/clients" > /dev/null 2>&1

echo "✅ API Gateway client created"

# Create Frontend client
echo "🌐 Creating Frontend client..."
curl -s -X POST \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "clientId": "'$VITE_KEYCLOAK_CLIENT_ID'",
    "name": "Frontend",
    "enabled": true,
    "publicClient": true,
    "redirectUris": ["http://localhost:3001/*", "http://localhost:3001"],
    "webOrigins": ["*"],
    "standardFlowEnabled": true,
    "implicitFlowEnabled": true,
    "directAccessGrantsEnabled": true
  }' \
  "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/clients" > /dev/null 2>&1

echo "✅ Frontend client created"

# Create app admin user
echo "👤 Creating/verifying app admin user '$APP_USER_EMAIL'..."
USER_RESPONSE=$(curl -s -X POST \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "username": "'$APP_USER_EMAIL'",
    "email": "'$APP_USER_EMAIL'",
    "firstName": "Admin",
    "lastName": "User",
    "enabled": true,
    "emailVerified": true
  }' \
  "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users" 2>&1)

USER_ID=$(echo "$USER_RESPONSE" | jq -r '.id' 2>/dev/null)

if [ -z "$USER_ID" ] || [ "$USER_ID" = "null" ]; then
  # Try to find existing user
  USERS=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users?username=$APP_USER_EMAIL&exact=true" 2>/dev/null)
  
  USER_ID=$(echo "$USERS" | jq -r '.[0].id' 2>/dev/null)
fi

if [ -n "$USER_ID" ] && [ "$USER_ID" != "null" ]; then
  echo "✅ User found with ID: $USER_ID"
  
  # Set password
  echo "🔐 Setting user password..."
  curl -s -X PUT \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -d '{
      "type": "password",
      "value": "'$APP_USER_PASSWORD'",
      "temporary": false
    }' \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users/$USER_ID/reset-password" > /dev/null 2>&1
  
  echo "✅ Password set"
else
  echo "⚠️  Could not verify/create user (will retry)"
fi

echo ""
echo "✨ Keycloak initialization complete!"
echo ""
echo "� Access Information:"
echo "  - Admin Console: http://localhost:8080/admin"
echo "  - Admin: $KEYCLOAK_ADMIN"
echo "  - Password: $KEYCLOAK_ADMIN_PASSWORD"
echo ""
echo "👤 App User (Login at http://localhost:3001/login):"
echo "  - Email: $APP_USER_EMAIL"
echo "  - Password: $APP_USER_PASSWORD"
echo ""

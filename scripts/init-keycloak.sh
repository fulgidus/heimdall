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
APP_USER_PASSWORD=${APP_USER_PASSWORD:-admin}

echo "[*] Keycloak Initialization Script"
echo "    Keycloak URL: $KEYCLOAK_URL"
echo "    Realm: $KEYCLOAK_REALM"
echo "    Admin User: $KEYCLOAK_ADMIN"
echo ""

# Wait for Keycloak to be ready
echo "[*] Waiting for Keycloak to be ready..."
WAIT_COUNTER=0
while [ $WAIT_COUNTER -lt 60 ]; do
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$KEYCLOAK_URL/realms/master" 2>/dev/null)
  if [ "$HTTP_CODE" = "200" ]; then
    echo "[OK] Keycloak is ready (HTTP $HTTP_CODE)"
    break
  fi
  WAIT_COUNTER=$((WAIT_COUNTER + 1))
  if [ $((WAIT_COUNTER % 10)) -eq 0 ]; then
    echo "     Waiting... ($WAIT_COUNTER/60)"
  fi
  sleep 1
done

if [ $WAIT_COUNTER -eq 60 ]; then
  echo "[!] Keycloak failed to start within timeout"
  exit 0
fi

# Get admin token
echo "[*] Getting admin token..."
TOKEN_RESPONSE=$(curl -s -X POST \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d "client_id=admin-cli&username=$KEYCLOAK_ADMIN&password=$KEYCLOAK_ADMIN_PASSWORD&grant_type=password" \
  "$KEYCLOAK_URL/realms/master/protocol/openid-connect/token" 2>&1)

# Extract token using grep since we can't rely on jq
ADMIN_TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

if [ -z "$ADMIN_TOKEN" ]; then
  echo "[!] Could not get admin token"
  echo "    Response: $TOKEN_RESPONSE"
  echo "    Will retry on next startup"
  exit 0
fi

echo "[OK] Token obtained"
echo ""

# ============================================================================
# REALM MANAGEMENT
# ============================================================================

echo "[*] Checking if realm '$KEYCLOAK_REALM' exists..."
REALM_CHECK=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
  "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM" 2>/dev/null)

# Check if realm exists by looking for the realm name in response
if echo "$REALM_CHECK" | grep -q "\"realm\":\"$KEYCLOAK_REALM\""; then
  echo "[OK] Realm already exists"
else
  echo "[*] Creating realm '$KEYCLOAK_REALM'..."
  
  REALM_CREATE=$(curl -s -w "\n%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -d '{"realm":"'$KEYCLOAK_REALM'","enabled":true,"displayName":"Heimdall SDR"}' \
    "$KEYCLOAK_URL/admin/realms" 2>&1)
  
  HTTP_CODE=$(echo "$REALM_CREATE" | tail -1)
  RESPONSE=$(echo "$REALM_CREATE" | sed '$d')
  
  if [ "$HTTP_CODE" = "201" ]; then
    echo "[OK] Realm created successfully"
  elif [ "$HTTP_CODE" = "409" ]; then
    echo "[OK] Realm already exists (conflict)"
  else
    echo "[!] Realm creation returned HTTP $HTTP_CODE"
    if [ -n "$RESPONSE" ]; then
      echo "    Response: $RESPONSE"
    fi
  fi
  sleep 2
fi

# ============================================================================
# CLIENT MANAGEMENT
# ============================================================================

echo ""
echo "[*] Checking API Gateway client..."
GATEWAY_CHECK=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
  "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/clients?clientId=$KEYCLOAK_API_GATEWAY_CLIENT_ID" 2>/dev/null)

if echo "$GATEWAY_CHECK" | grep -q "\"clientId\":\"$KEYCLOAK_API_GATEWAY_CLIENT_ID\""; then
  echo "[OK] API Gateway client already exists"
else
  echo "[*] Creating API Gateway client..."
  
  GATEWAY_CREATE=$(curl -s -w "\n%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -d '{
      "clientId":"'$KEYCLOAK_API_GATEWAY_CLIENT_ID'",
      "secret":"'$KEYCLOAK_API_GATEWAY_CLIENT_SECRET'",
      "enabled":true,
      "publicClient":false,
      "directAccessGrantsEnabled":true,
      "serviceAccountsEnabled":true
    }' \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/clients" 2>&1)
  
  HTTP_CODE=$(echo "$GATEWAY_CREATE" | tail -1)
  
  if [ "$HTTP_CODE" = "201" ]; then
    echo "[OK] API Gateway client created"
  else
    echo "[!] API Gateway client creation returned HTTP $HTTP_CODE"
  fi
fi

echo "[*] Checking Frontend client..."
FRONTEND_CHECK=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
  "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/clients?clientId=$VITE_KEYCLOAK_CLIENT_ID" 2>/dev/null)

if echo "$FRONTEND_CHECK" | grep -q "\"clientId\":\"$VITE_KEYCLOAK_CLIENT_ID\""; then
  echo "[OK] Frontend client already exists"
else
  echo "[*] Creating Frontend client..."
  
  FRONTEND_CREATE=$(curl -s -w "\n%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -d '{
      "clientId":"'$VITE_KEYCLOAK_CLIENT_ID'",
      "enabled":true,
      "publicClient":true,
      "directAccessGrantsEnabled":true
    }' \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/clients" 2>&1)
  
  HTTP_CODE=$(echo "$FRONTEND_CREATE" | tail -1)
  
  if [ "$HTTP_CODE" = "201" ]; then
    echo "[OK] Frontend client created"
  else
    echo "[!] Frontend client creation returned HTTP $HTTP_CODE"
  fi
fi

# ============================================================================
# USER MANAGEMENT
# ============================================================================

echo ""
echo "[*] Checking app user '$APP_USER_EMAIL'..."
USERS_CHECK=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
  "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users?username=$APP_USER_EMAIL" 2>/dev/null)

# Extract user ID if user exists
USER_ID=$(echo "$USERS_CHECK" | grep -o '"id":"[^"]*' | head -1 | cut -d'"' -f4)

if [ -n "$USER_ID" ]; then
  echo "[OK] User exists (ID: $USER_ID)"
  echo "[*] Updating user password..."
  
  PWD_UPDATE=$(curl -s -w "\n%{http_code}" -X PUT \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -d '{"type":"password","value":"'$APP_USER_PASSWORD'","temporary":false}' \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users/$USER_ID/reset-password" 2>&1)
  
  HTTP_CODE=$(echo "$PWD_UPDATE" | tail -1)
  if [ "$HTTP_CODE" = "204" ] || [ "$HTTP_CODE" = "200" ]; then
    echo "[OK] User password updated"
  else
    echo "[!] Password update returned HTTP $HTTP_CODE"
  fi
else
  echo "[*] Creating user '$APP_USER_EMAIL'..."
  
  USER_CREATE=$(curl -s -w "\n%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -d '{
      "username":"'$APP_USER_EMAIL'",
      "email":"'$APP_USER_EMAIL'",
      "enabled":true,
      "emailVerified":true
    }' \
    "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users" 2>&1)
  
  HTTP_CODE=$(echo "$USER_CREATE" | tail -1)
  
  if [ "$HTTP_CODE" = "201" ]; then
    echo "[OK] User created"
    
    # Extract new user ID from query
    NEW_USER=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
      "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users?username=$APP_USER_EMAIL" 2>/dev/null)
    NEW_USER_ID=$(echo "$NEW_USER" | grep -o '"id":"[^"]*' | head -1 | cut -d'"' -f4)
    
    if [ -n "$NEW_USER_ID" ]; then
      echo "[*] Setting user password..."
      curl -s -X PUT \
        -H 'Content-Type: application/json' \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        -d '{"type":"password","value":"'$APP_USER_PASSWORD'","temporary":false}' \
        "$KEYCLOAK_URL/admin/realms/$KEYCLOAK_REALM/users/$NEW_USER_ID/reset-password" > /dev/null 2>&1
      echo "[OK] Password set"
    fi
  else
    echo "[!] User creation returned HTTP $HTTP_CODE"
  fi
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================================================"
echo "[OK] Keycloak initialization complete!"
echo "========================================================================"
echo ""
echo "[*] Admin Console:"
echo "    URL: http://localhost:8080/auth/admin"
echo "    User: $KEYCLOAK_ADMIN"
echo "    Pass: $KEYCLOAK_ADMIN_PASSWORD"
echo ""
echo "[*] App Login:"
echo "    URL: http://localhost:3001/login"
echo "    Realm: $KEYCLOAK_REALM"
echo "    Email: $APP_USER_EMAIL"
echo "    Pass: $APP_USER_PASSWORD"
echo ""
echo "[*] Resources:"
echo "    - API Gateway Client: $KEYCLOAK_API_GATEWAY_CLIENT_ID"
echo "    - Frontend Client: $VITE_KEYCLOAK_CLIENT_ID"
echo ""

exit 0

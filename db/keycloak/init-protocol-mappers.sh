#!/bin/bash
#
# Keycloak Protocol Mappers Initialization Script
# 
# This script adds protocol mappers to the heimdall-frontend client
# to ensure JWT tokens include realm_access.roles and user claims.
#
# Usage: ./init-protocol-mappers.sh
#
# Prerequisites:
# - Keycloak must be running and healthy
# - Default admin credentials: admin/admin
#

set -e

KEYCLOAK_URL="${KEYCLOAK_URL:-http://localhost:8080}"
KEYCLOAK_ADMIN_USER="${KEYCLOAK_ADMIN_USER:-admin}"
KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD:-admin}"
REALM_NAME="${REALM_NAME:-heimdall}"
CLIENT_ID="${CLIENT_ID:-heimdall-frontend}"

echo "🔐 Keycloak Protocol Mappers Initialization"
echo "==========================================="
echo "Keycloak URL: $KEYCLOAK_URL"
echo "Realm: $REALM_NAME"
echo "Client: $CLIENT_ID"
echo ""

# Step 1: Get admin access token
echo "📝 Step 1: Authenticating as admin..."
ADMIN_TOKEN=$(curl -s -X POST "$KEYCLOAK_URL/realms/master/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=admin-cli" \
  -d "username=$KEYCLOAK_ADMIN_USER" \
  -d "password=$KEYCLOAK_ADMIN_PASSWORD" \
  -d "grant_type=password" | jq -r '.access_token')

if [ -z "$ADMIN_TOKEN" ] || [ "$ADMIN_TOKEN" = "null" ]; then
  echo "❌ Failed to authenticate. Check Keycloak credentials."
  exit 1
fi
echo "✅ Authentication successful"

# Step 2: Get client internal ID
echo ""
echo "📝 Step 2: Finding client ID for '$CLIENT_ID'..."
INTERNAL_CLIENT_ID=$(curl -s "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq -r ".[] | select(.clientId == \"$CLIENT_ID\") | .id")

if [ -z "$INTERNAL_CLIENT_ID" ] || [ "$INTERNAL_CLIENT_ID" = "null" ]; then
  echo "❌ Client '$CLIENT_ID' not found in realm '$REALM_NAME'"
  exit 1
fi
echo "✅ Found client: $INTERNAL_CLIENT_ID"

# Step 3: Check existing protocol mappers
echo ""
echo "📝 Step 3: Checking existing protocol mappers..."
EXISTING_MAPPERS=$(curl -s "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients/$INTERNAL_CLIENT_ID/protocol-mappers/models" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq 'length')
echo "Found $EXISTING_MAPPERS existing protocol mapper(s)"

# Step 4: Add protocol mappers
echo ""
echo "📝 Step 4: Adding protocol mappers..."

# Add realm-roles mapper
echo "  ➤ Adding 'realm-roles' mapper..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients/$INTERNAL_CLIENT_ID/protocol-mappers/models" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "realm-roles",
    "protocol": "openid-connect",
    "protocolMapper": "oidc-usermodel-realm-role-mapper",
    "consentRequired": false,
    "config": {
      "claim.name": "realm_access.roles",
      "jsonType.label": "String",
      "multivalued": "true",
      "access.token.claim": "true",
      "id.token.claim": "true",
      "userinfo.token.claim": "true"
    }
  }')
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" -eq 201 ] || [ "$HTTP_CODE" -eq 204 ] || [ "$HTTP_CODE" -eq 409 ]; then
  echo "    ✅ realm-roles mapper added (HTTP $HTTP_CODE)"
else
  echo "    ⚠️  realm-roles mapper may already exist or failed (HTTP $HTTP_CODE)"
fi

# Add email mapper
echo "  ➤ Adding 'email' mapper..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients/$INTERNAL_CLIENT_ID/protocol-mappers/models" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "email",
    "protocol": "openid-connect",
    "protocolMapper": "oidc-usermodel-property-mapper",
    "consentRequired": false,
    "config": {
      "userinfo.token.claim": "true",
      "user.attribute": "email",
      "id.token.claim": "true",
      "access.token.claim": "true",
      "claim.name": "email",
      "jsonType.label": "String"
    }
  }')
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" -eq 201 ] || [ "$HTTP_CODE" -eq 204 ] || [ "$HTTP_CODE" -eq 409 ]; then
  echo "    ✅ email mapper added (HTTP $HTTP_CODE)"
else
  echo "    ⚠️  email mapper may already exist or failed (HTTP $HTTP_CODE)"
fi

# Add name mapper
echo "  ➤ Adding 'name' mapper..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients/$INTERNAL_CLIENT_ID/protocol-mappers/models" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "name",
    "protocol": "openid-connect",
    "protocolMapper": "oidc-full-name-mapper",
    "consentRequired": false,
    "config": {
      "id.token.claim": "true",
      "access.token.claim": "true",
      "userinfo.token.claim": "true"
    }
  }')
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" -eq 201 ] || [ "$HTTP_CODE" -eq 204 ] || [ "$HTTP_CODE" -eq 409 ]; then
  echo "    ✅ name mapper added (HTTP $HTTP_CODE)"
else
  echo "    ⚠️  name mapper may already exist or failed (HTTP $HTTP_CODE)"
fi

# Add preferred_username mapper
echo "  ➤ Adding 'preferred_username' mapper..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients/$INTERNAL_CLIENT_ID/protocol-mappers/models" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "preferred_username",
    "protocol": "openid-connect",
    "protocolMapper": "oidc-usermodel-property-mapper",
    "consentRequired": false,
    "config": {
      "userinfo.token.claim": "true",
      "user.attribute": "username",
      "id.token.claim": "true",
      "access.token.claim": "true",
      "claim.name": "preferred_username",
      "jsonType.label": "String"
    }
  }')
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" -eq 201 ] || [ "$HTTP_CODE" -eq 204 ] || [ "$HTTP_CODE" -eq 409 ]; then
  echo "    ✅ preferred_username mapper added (HTTP $HTTP_CODE)"
else
  echo "    ⚠️  preferred_username mapper may already exist or failed (HTTP $HTTP_CODE)"
fi

# Step 5: Verify protocol mappers
echo ""
echo "📝 Step 5: Verifying protocol mappers..."
FINAL_MAPPERS=$(curl -s "$KEYCLOAK_URL/admin/realms/$REALM_NAME/clients/$INTERNAL_CLIENT_ID/protocol-mappers/models" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq 'length')
echo "✅ Client now has $FINAL_MAPPERS protocol mapper(s)"

echo ""
echo "🎉 Protocol mappers initialization complete!"
echo ""
echo "Test with:"
echo "  curl -s -X POST \"$KEYCLOAK_URL/realms/$REALM_NAME/protocol/openid-connect/token\" \\"
echo "    -d 'client_id=$CLIENT_ID' \\"
echo "    -d 'username=admin@heimdall.local' \\"
echo "    -d 'password=admin' \\"
echo "    -d 'grant_type=password' | jq -r '.access_token' | cut -d'.' -f2 | base64 -d 2>/dev/null | jq '.realm_access.roles'"

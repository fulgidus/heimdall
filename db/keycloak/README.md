# Keycloak Configuration

This directory contains Keycloak realm configuration and initialization scripts for the Heimdall project.

## Files

- **`heimdall-realm.json`**: Keycloak realm export defining users, roles, clients, and settings
- **`init-protocol-mappers.sh`**: Script to add protocol mappers to the `heimdall-frontend` client (see Known Issues below)

## Realm Configuration

The `heimdall-realm.json` file defines:

### Users
- **admin@heimdall.local** (password: `admin`) - Full system access
- **operator@heimdall.local** (password: `operator`) - Operator role
- **viewer@heimdall.local** (password: `viewer`) - Read-only access

### Roles
- `admin` - Full system administrator
- `operator` - Can create/edit constellations and datasets
- `viewer` - Read-only access to public resources

### Clients
- **heimdall-frontend** - React frontend (public client)
- **api-gateway** - Backend API Gateway (confidential)
- **rf-acquisition** - RF data collection service (confidential)
- **training** - ML training service (confidential)
- **inference** - Real-time inference service (confidential)

## Quick Start

### 1. Start Keycloak

```bash
docker compose up -d keycloak
```

The realm will be automatically imported on first startup.

### 2. Initialize Protocol Mappers (Required on First Run)

Due to a Keycloak realm import limitation, protocol mappers must be added after the first startup:

```bash
./db/keycloak/init-protocol-mappers.sh
```

This script adds the necessary protocol mappers to include `realm_access.roles`, `email`, `name`, and `preferred_username` claims in JWT tokens.

### 3. Verify Authentication

Test login and verify JWT token contains roles:

```bash
curl -s -X POST "http://localhost:8080/realms/heimdall/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=heimdall-frontend" \
  -d "username=admin@heimdall.local" \
  -d "password=admin" \
  -d "grant_type=password" | jq -r '.access_token' | cut -d'.' -f2 | base64 -d 2>/dev/null | jq '.'
```

Expected output should include:

```json
{
  "realm_access": {
    "roles": ["admin"]
  },
  "email": "admin@heimdall.local",
  "name": "Admin User",
  "preferred_username": "admin"
}
```

## Known Issues

### Protocol Mappers Not Imported

**Problem**: Keycloak 23.x has a known issue where protocol mappers defined in `protocolMappers` array within client definitions are not imported via `--import-realm` flag.

**Symptoms**:
- JWT tokens missing `realm_access.roles` claim
- Frontend shows "Access Denied" even for admin users
- Login succeeds but user role is undefined

**Solution**: Run `./db/keycloak/init-protocol-mappers.sh` after first Keycloak startup to add protocol mappers via Admin API.

**Workaround for Fresh Installs**:

```bash
# Stop Keycloak
docker compose stop keycloak

# Remove data volume to force re-import
docker volume rm heimdall_keycloak_data

# Start Keycloak
docker compose up -d keycloak

# Wait 30s for Keycloak to become healthy
sleep 30

# Initialize protocol mappers
./db/keycloak/init-protocol-mappers.sh
```

### CORS Configuration

The realm is configured to allow CORS from:
- `http://localhost:3000` (production frontend)
- `http://localhost:3001` (alternative port)
- `http://localhost:5173` (Vite dev server)
- `http://localhost:8000` (backend)

If you need to add additional origins, update the `KC_SPI_CORS_ORIGINS` environment variable in `docker-compose.yml`.

## Admin Console

Access Keycloak Admin Console at: **http://localhost:8080**

- Username: `admin`
- Password: `admin` (change in production via `KEYCLOAK_ADMIN_PASSWORD` env var)

## Security Notes

⚠️ **Production Checklist**:

1. Change default admin password:
   ```bash
   KEYCLOAK_ADMIN_PASSWORD=<strong-password> docker compose up -d keycloak
   ```

2. Change client secrets in `heimdall-realm.json` for:
   - `api-gateway-secret-change-in-production`
   - `rf-acquisition-secret-change-in-production`
   - `training-secret-change-in-production`
   - `inference-secret-change-in-production`

3. Change default user passwords (`admin`, `operator`, `viewer`)

4. Enable HTTPS and disable `KC_HTTP_ENABLED=true`

5. Set proper `KC_HOSTNAME` for production domain

6. Review and restrict CORS origins in `KC_SPI_CORS_ORIGINS`

## Troubleshooting

### "Invalid credentials" on login

- Verify Keycloak is healthy: `docker compose ps keycloak`
- Check realm import succeeded: Look for "Realm 'heimdall' - data exported" in logs
- Verify user exists in Admin Console

### "Access Denied" for admin user

- Run `init-protocol-mappers.sh` to add protocol mappers
- Verify JWT token includes `realm_access.roles` claim (see verification command above)
- Check browser console for authentication errors

### Changes to realm.json not reflected

Keycloak only imports realm on **first startup**. To apply changes:

```bash
docker compose stop keycloak
docker volume rm heimdall_keycloak_data
docker compose up -d keycloak
sleep 30
./db/keycloak/init-protocol-mappers.sh
```

## References

- [Keycloak Documentation](https://www.keycloak.org/documentation)
- [Keycloak Docker Image](https://quay.io/repository/keycloak/keycloak)
- [OpenID Connect Specification](https://openid.net/specs/openid-connect-core-1_0.html)
- [Heimdall RBAC Documentation](../../docs/RBAC.md)

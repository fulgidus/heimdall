# 🌐 API URL Standards - Frontend

**Last Updated**: 2025-11-01  
**Purpose**: Prevent double `/api/api` path issues

---

## ✅ Correct Pattern

### Base URL Configuration (`src/lib/api.ts`)

```typescript
const API_BASE_URL = `${protocol}//${host}/api`;
// Example: http://localhost/api
```

### API Endpoint Definitions

**ALL endpoint paths MUST start with `/v1/`** (NOT `/api/v1/`)

```typescript
// ✅ CORRECT
const response = await api.get('/v1/acquisition/websdrs');
const response = await api.post('/v1/sessions', data);
const response = await api.delete(`/v1/websdrs/${id}`);

// ❌ WRONG - causes /api/api/v1/... double prefix
const response = await api.get('/api/v1/acquisition/websdrs');
```

---

## 📋 Complete URL Construction

```
baseURL        +  endpoint path           =  final URL
/api           +  /v1/acquisition/websdrs =  /api/v1/acquisition/websdrs
http://host/api + /v1/sessions           =  http://host/api/v1/sessions
```

---

## 🔍 How to Verify

### Search for violations:

```bash
# Find any /api/v1/ references in source code
cd frontend
grep -r "/api/v1/" src/ --include="*.ts" --include="*.tsx"

# Should return 0 results
```

### Check specific files:

```bash
# API service files
grep "/api/v1/" src/services/api/*.ts

# Store files
grep "/api/v1/" src/store/*.ts

# Component files
grep "/api/v1/" src/components/*.tsx src/pages/*.tsx
```

---

## 🛠️ Fixing Violations

### Automated fix:

```bash
cd frontend

# Fix single quotes
find src -name "*.ts" -o -name "*.tsx" | xargs sed -i "s|'/api/v1/|'/v1/|g"

# Fix double quotes
find src -name "*.ts" -o -name "*.tsx" | xargs sed -i 's|"/api/v1/|"/v1/|g'

# Fix template literals
find src -name "*.ts" -o -name "*.tsx" | xargs sed -i 's|`/api/v1/|`/v1/|g'
```

### Manual fix examples:

```typescript
// BEFORE
export async function getWebSDRs() {
  return api.get('/api/v1/acquisition/websdrs');
}

// AFTER
export async function getWebSDRs() {
  return api.get('/v1/acquisition/websdrs');
}
```

---

## 🧪 Test Mock Configuration

Test mocks MUST also use `/v1/` paths:

```typescript
// ✅ CORRECT
mock.onGet('/v1/websdrs').reply(200, mockData);

// ❌ WRONG
mock.onGet('/api/v1/websdrs').reply(200, mockData);
```

---

## 📦 Files to Update When Adding New Endpoints

1. **Service file** (`src/services/api/*.ts`): Use `/v1/...` paths
2. **Test file** (`src/services/api/*.test.ts`): Mock with `/v1/...` paths
3. **Store** (if applicable): Use service functions (no direct URLs)
4. **Components**: Use service functions (no direct URLs)

---

## 🚨 CI/CD Pre-commit Check

Add to `.husky/pre-commit` or CI pipeline:

```bash
#!/bin/bash
# Check for /api/v1/ violations in frontend
if grep -r "/api/v1/" frontend/src/ --include="*.ts" --include="*.tsx" -q; then
  echo "❌ ERROR: Found /api/v1/ in source code. Use /v1/ instead."
  echo "Run: cd frontend && find src -name '*.ts' -o -name '*.tsx' | xargs sed -i \"s|'/api/v1/|'/v1/|g\""
  exit 1
fi
echo "✅ No /api/v1/ violations found"
```

---

## 📚 Related Documentation

- [`docs/CORS_CONFIGURATION.md`](../docs/CORS_CONFIGURATION.md) - CORS and API Gateway setup
- [`frontend/src/lib/api.ts`](src/lib/api.ts) - Axios client configuration
- [`db/envoy/envoy.yaml`](../db/envoy/envoy.yaml) - Envoy proxy routing

---

**Remember**: 
- `baseURL = /api` (configured in `api.ts`)
- `endpoint paths = /v1/...` (no `/api` prefix)
- `final URL = /api/v1/...` (concatenated by axios)

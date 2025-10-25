# Multi-stage build for frontend
# NOTE: There is a known issue with npm 10.8.2 in Docker (node:20-slim) that causes 
# "Exit handler never called" error and prevents devDependencies from installing properly.
# 
# WORKAROUNDS:
# 1. Build frontend locally first, then copy dist/ folder (fastest)
# 2. Use docker-compose which mounts local node_modules
# 3. Wait for npm bug fix: https://github.com/npm/cli/issues
#
# For now, this Dockerfile requires a pre-built dist/ folder.
# Build locally first with: npm run build

FROM nginx:1.25-alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf /etc/nginx/conf.d/default.conf

# Copy pre-built frontend (build locally first with: npm run build)
COPY dist /usr/share/nginx/html

# Generate build info
RUN echo "Build: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" > /usr/share/nginx/html/BUILD_INFO.txt

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost:80/ || exit 1

# Document port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

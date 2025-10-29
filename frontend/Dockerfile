# Build stage
FROM node:20 AS builder

WORKDIR /app

# Copy everything first (simpler approach for now)
COPY . .

# Build arguments for environment variables (can be overridden at build time)
ARG VITE_API_URL=http://localhost:8000
ARG VITE_API_TIMEOUT=10000
ARG VITE_ENV=production
ARG VITE_ENABLE_ANALYTICS=true
ARG VITE_ENABLE_DEBUG=false
ARG VITE_KEYCLOAK_URL=http://localhost:8080
ARG VITE_KEYCLOAK_REALM=heimdall
ARG VITE_KEYCLOAK_CLIENT_ID=heimdall-frontend
ARG VITE_ADMIN_EMAIL=admin@heimdall.local
ARG VITE_ADMIN_PASSWORD=admin
ARG VITE_AUTH_TOKEN_KEY=heimdall_auth_token
ARG VITE_SESSION_TIMEOUT=3600000
ARG VITE_SOCKET_URL=ws://localhost:8000/ws
ARG VITE_MAPBOX_TOKEN

# Set environment variables for build
ENV VITE_API_URL=$VITE_API_URL \
    VITE_API_TIMEOUT=$VITE_API_TIMEOUT \
    VITE_ENV=$VITE_ENV \
    VITE_ENABLE_ANALYTICS=$VITE_ENABLE_ANALYTICS \
    VITE_ENABLE_DEBUG=$VITE_ENABLE_DEBUG \
    VITE_KEYCLOAK_URL=$VITE_KEYCLOAK_URL \
    VITE_KEYCLOAK_REALM=$VITE_KEYCLOAK_REALM \
    VITE_KEYCLOAK_CLIENT_ID=$VITE_KEYCLOAK_CLIENT_ID \
    VITE_ADMIN_EMAIL=$VITE_ADMIN_EMAIL \
    VITE_ADMIN_PASSWORD=$VITE_ADMIN_PASSWORD \
    VITE_AUTH_TOKEN_KEY=$VITE_AUTH_TOKEN_KEY \
    VITE_SESSION_TIMEOUT=$VITE_SESSION_TIMEOUT \
    VITE_SOCKET_URL=$VITE_SOCKET_URL \
    VITE_MAPBOX_TOKEN=$VITE_MAPBOX_TOKEN

# Install dependencies and build
RUN npm install --legacy-peer-deps && npm run build

# Runtime stage - Using nginx for better performance
FROM nginx:alpine

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built application from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Create a script to inject runtime environment variables
RUN echo $'#!/bin/sh\n\
    set -e\n\
    \n\
    # Set default values if not provided\n\
    export VITE_API_URL=${VITE_API_URL:-http://localhost:8000}\n\
    export VITE_ENV=${VITE_ENV:-production}\n\
    \n\
    # Substitute variables in nginx.conf\n\
    envsubst \'$VITE_API_URL $VITE_ENV\' < /etc/nginx/nginx.conf > /etc/nginx/nginx.conf.new && mv /etc/nginx/nginx.conf.new /etc/nginx/nginx.conf\n\
    \n\
    # Replace environment variables in JavaScript files\n\
    for file in /usr/share/nginx/html/assets/*.js; do\n\
    if [ -f "$file" ]; then\n\
    # Create backup\n\
    cp "$file" "$file.bak"\n\
    \n\
    # Replace environment variables\n\
    envsubst "\\$VITE_API_URL \\$VITE_ENV" < "$file.bak" > "$file"\n\
    rm "$file.bak"\n\
    fi\n\
    done\n\
    \n\
    # Start nginx\n\
    exec nginx -g "daemon off;"' > /docker-entrypoint.sh

RUN chmod +x /docker-entrypoint.sh

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:3000/ || exit 1

# Start nginx with environment variable injection
ENTRYPOINT ["/docker-entrypoint.sh"]

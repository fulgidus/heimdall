# Stage 1: Build
FROM node:20-alpine as builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies with exact versions
RUN npm ci --legacy-peer-deps

# Copy source
COPY . .

# Build frontend
RUN npm run build

# Generate build info
RUN echo "Build: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" > dist/BUILD_INFO.txt

# Stage 2: Serve
FROM nginx:1.25-alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf /etc/nginx/conf.d/default.conf

# Copy built frontend from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost:80/ || exit 1

# Document port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

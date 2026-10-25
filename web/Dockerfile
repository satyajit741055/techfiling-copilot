# =============================================================================
# Frontend Dockerfile - TechFiling Copilot (Next.js standalone)
# =============================================================================

FROM node:24-alpine AS deps

# Recommended for Next.js native binaries on Alpine (musl)
RUN apk add --no-cache libc6-compat

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

# =============================================================================
# Builder stage
# =============================================================================
FROM node:24-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# The browser only calls this app's /api/* proxy; BACKEND_URL and API_ACCESS_KEY
# are read at runtime on the server (see docker-compose.yml), not baked in here.
ENV NEXT_TELEMETRY_DISABLED=1

# public/ is empty and untracked by git, so make sure it exists in fresh clones
RUN mkdir -p public && npm run build

# =============================================================================
# Runner stage
# =============================================================================
FROM node:24-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs

COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

# wget is provided by busybox in the Alpine base image
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:3000 || exit 1

CMD ["node", "server.js"]

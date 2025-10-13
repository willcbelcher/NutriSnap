# --- STAGE 1: Dependency & Build (using yarn) ---
# Use a Node image for building, which contains necessary tools (like compilers)
FROM node:20-alpine AS build

# Set build arguments (can be overridden during docker build)
ARG APP_DIR="/app"
ENV APP_DIR=${APP_DIR}

# Create a non-root user and group for security best practice
RUN addgroup --system nuxtuser && adduser --system -G nuxtuser nuxtuser

# Set the working directory
WORKDIR ${APP_DIR}

# Copy package files first to leverage Docker layer caching
# NOTE: Using yarn.lock for Yarn dependency management.
COPY package.json yarn.lock ./

# Install dependencies (as root, which is safe in this build stage)
# Using --frozen-lockfile ensures deterministic builds based on yarn.lock
RUN yarn install --frozen-lockfile

# Copy the rest of the application source code
COPY . .

# Build the project
RUN yarn build

# Build Stage 2

FROM node:22-alpine
WORKDIR /app

# Only `.output` folder is needed from the build stage
COPY --from=build /app/.output/ ./

# Change the port and host
ENV PORT=80
ENV HOST=0.0.0.0

EXPOSE 80

CMD ["node", "/app/server/index.mjs"]

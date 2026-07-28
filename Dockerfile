# Default to a standard python image, but can be overridden with a CUDA base if needed
# Example: --build-arg BASE_IMAGE=nvidia/cuda:12.2.0-base-ubuntu22.04
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

# Allow passing the extras (e.g. "[cuda12,qdax,rl]") during build
# If not specified, installs cpu backend
ARG EXTRAS="[cpu]"

# Install system dependencies
# git is required for some pip packages and fetching dependencies
# build-essential might be required by some pip packages compiling C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# If using an NVIDIA base image, we may need to install python and pip
# We check if python3 exists, if not we install it
RUN command -v python3 || (apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev && rm -rf /var/lib/apt/lists/*)
# Ensure python aliases to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python || true

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy the minimum files needed for dependency resolution to cache the layer
COPY pyproject.toml README.md ./
# We need the __init__.py so hatchling can read the version/description if configured that way
RUN mkdir -p src/malthusjax
COPY src/malthusjax/__init__.py ./src/malthusjax/__init__.py

# Install dependencies based on EXTRAS
RUN python -m pip install --no-cache-dir -e ".${EXTRAS}"

# Copy the rest of the source code
COPY . .

# ==============================================================================
# JAX/XLA MEMORY MANAGEMENT (OPTIONAL)
# ==============================================================================
# JAX defaults to aggressively pre-allocating 90% of GPU VRAM. If you are 
# running this container on a shared cluster GPU, you will likely encounter 
# Out-Of-Memory (OOM) errors or block other users.
#
# Uncomment the following lines to use on-demand memory allocation, or pass 
# them at runtime via `docker run -e XLA_PYTHON_CLIENT_PREALLOCATE=false ...`
#
# ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
# ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
# ==============================================================================

# Run the mjax CLI by default
ENTRYPOINT ["mjax"]
CMD ["--help"]

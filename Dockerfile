FROM --platform=linux/amd64 ghcr.io/prefix-dev/pixi:0.43.3

# Work directory
WORKDIR /workspace

# Copy environment definition first (helps caching)
COPY pyproject.toml pixi.lock ./

# Utilities
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends build-essential \
    sudo curl git htop less rsync screen vim nano wget ca-certificates \
    openssh-client zsh clang graphviz

# Now copy the actual project
COPY . .

# Install dependencies (will be cached unless lock changes)
RUN pixi install --frozen

# Default: open pixi environment shell
# (You can override when calling `docker run`)
ENTRYPOINT ["pixi", "run"]


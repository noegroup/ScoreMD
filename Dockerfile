FROM --platform=linux/amd64 python:3.12-slim AS linux-base

# Utilities
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends build-essential \
    sudo curl git htop less rsync screen vim nano wget ca-certificates \
    openssh-client zsh clang graphviz

# Download and install VS Code Server CLI
# This is only for development purposes
RUN wget -O /tmp/vscode-server-cli.tar.gz "https://update.code.visualstudio.com/latest/cli-linux-x64/stable" && \
    mkdir -p /usr/local/bin && \
    tar -xf /tmp/vscode-server-cli.tar.gz -C /usr/local/bin && \
    rm /tmp/vscode-server-cli.tar.gz

# Slurm
RUN COMMANDS="sacct sacctmgr salloc sattach sbatch sbcast scancel scontrol sdiag sgather sinfo smap sprio squeue sreport srun sshare sstat strigger sview" \
    && for CMD in $COMMANDS; do echo '#!/bin/bash' > "/usr/local/bin/$CMD" \
    && echo 'ssh $USER@$SLURM_CLUSTER_NAME -t "cd $PWD; . ~/.zshrc 2>/dev/null || . ~/.bashrc 2>/dev/null; bash -lc '\'$CMD \$@\''"' >> "/usr/local/bin/$CMD" \
    && chmod +x "/usr/local/bin/$CMD"; done

FROM linux-base AS python-base

# Workdir
WORKDIR /srv/repo

# Install pixi
COPY --from=ghcr.io/prefix-dev/pixi:0.43.3 /usr/local/bin/pixi /usr/local/bin/pixi

# we need to set these environment variables to let apptainer play nicely with the existing operating system
ENV PIXI_CACHE_DIR="/srv/.cache/rattler"
ENV UV_CACHE_DIR="/srv/.cache/uv"
ENV PIXI_HOME="/srv/.pixi_home"
ENV PIXI_ENVIRONMENT="/srv/.pixi_env"

RUN mkdir -p ${PIXI_HOME} ${PIXI_ENVIRONMENT}
RUN echo "detached-environments = \"${PIXI_ENVIRONMENT}\"" > ${PIXI_HOME}/config.toml

# This is a hack because pixi tries to install the editable dependencies
RUN mkdir -p src/ffdiffusion
RUN touch src/ffdiffusion/__init__.py

# Install dependencies
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=pixi.lock,target=pixi.lock \
    pixi install --frozen

RUN rm -rf src

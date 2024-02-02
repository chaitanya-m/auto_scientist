#!/bin/bash

# Default values for optional arguments
CONTAINER_NAME="lde_instance"
SSH_PATH="$HOME/.ssh"
GITCONFIG_PATH="$HOME/.gitconfig"

# Function to display usage
usage() {
    echo "Usage: $0 [container name] [ssh path] [gitconfig path]"
    echo " - Container name: Name of the Docker container (default: lde_instance)"
    echo " - SSH path: Path to the SSH directory (default: $HOME/.ssh)"
    echo " - Gitconfig path: Path to the .gitconfig file (default: $HOME/.gitconfig)"
}

# Assign arguments to variables, if provided
if [ "$#" -ge 1 ]; then
    CONTAINER_NAME=$1
fi
if [ "$#" -ge 2 ]; then
    SSH_PATH=$2
fi
if [ "$#" -ge 3 ]; then
    GITCONFIG_PATH=$3
fi

# Build the Docker image
docker build -t linux-dev-env .


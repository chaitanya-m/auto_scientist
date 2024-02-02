#!/bin/bash

# Default container name
CONTAINER_NAME="dev0"

# Check for an optional container name argument
if [ "$#" -eq 1 ]; then
    CONTAINER_NAME=$1
fi

# Run the Docker container in detached mode
docker run -d -v "$HOME/.ssh:/root/.ssh" -v "$HOME/.gitconfig:/root/.gitconfig" --name "$CONTAINER_NAME" linux-dev-env

# Attach to the container interactively
docker exec -it "$CONTAINER_NAME" /bin/bash


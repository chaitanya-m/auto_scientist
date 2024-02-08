#!/bin/bash

# Default container name
CONTAINER_NAME="dev0"

# Check for an optional container name argument
if [ "$#" -eq 1 ]; then
    CONTAINER_NAME=$1
fi

# Check if the container exists
container_exists=$(docker ps -a -q -f name=^/${CONTAINER_NAME}$)
if [ ! -z "$container_exists" ]; then
    echo "Container $CONTAINER_NAME exists."
    # Check if the container is running
    if [ $(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME) = "false" ]; then
        echo "Container $CONTAINER_NAME is stopped. Starting it..."
        docker start $CONTAINER_NAME
    else
        echo "Container $CONTAINER_NAME is already running."
    fi
else
    echo "Container $CONTAINER_NAME does not exist. Creating and starting it..."
    # Run the Docker container in detached mode
    docker run -d -v "$HOME/.ssh:/root/.ssh" -v "$HOME/.gitconfig:/root/.gitconfig" --name "$CONTAINER_NAME" linux-dev-env
fi

# Attach to the container interactively
docker exec -it "$CONTAINER_NAME" /bin/bash


#!/bin/bash
docker run -it -v ~/.ssh:/root/.ssh -v ~/.gitconfig:/root/.gitconfig --name lde_instance linux-dev-env


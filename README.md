
# auto_scientist Project Docker Environment

Welcome to the auto_scientist project Docker environment.

## Using the docker_build_run_lde.sh Script

Use the `docker_build_run_lde.sh` script to build and run this Docker container. This script allows specifying the container name, SSH path, and Gitconfig path as arguments.

### Usage

- To run with default settings:
  ```bash
  ./docker_build_run_lde.sh

- To specify a custom container name:
  ```bash
  ./docker_build_run_lde.sh my_custom_container_name

- To specify custom container name, SSH path, and Gitconfig path:
  ```bash
  ./docker_build_run_lde.sh my_custom_container_name /path/to/ssh /path/to/gitconfig

- Default SSH path: $HOME/.ssh

- Default Gitconfig path: $HOME/.gitconfig

- Refer to the script for more detailed usage instructions.

## Post-Setup

After running the container, you can clone the repository within the Docker environment using your own SSH credentials, ensuring secure access to the repository.

To clone the repository, use the following command in the terminal within the Docker container:

```bash
git clone git@github.com:chaitanya-m/auto_scientist.git

OR

```bash
chmod +x get_repo.sh
./get_repo.sh


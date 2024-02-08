
# auto_scientist Project Docker Environment

Welcome to the auto_scientist project Docker environment.

## Using the build_docker.sh and run_docker.sh Scripts

Use the `build_docker.sh` script to build this Docker container. This script allows specifying the container name, SSH path, and Gitconfig path as arguments.

'run_docker.sh' takes a container name as optional argument and otherwise defaults to dev0.

### Usage

- To run with default settings:
  ```bash
  ./build_docker.sh

- To specify a custom container name:
  ```bash
  ./build_docker.sh my_custom_container_name

- To specify custom container name, SSH path, and Gitconfig path:
  ```bash
  ./build_docker.sh my_custom_container_name /path/to/ssh /path/to/gitconfig

- Default SSH path: $HOME/.ssh

- Default Gitconfig path: $HOME/.gitconfig

- Refer to the scripts for more detailed usage instructions.

## Post-Setup

After running the container, you can clone the repository within the Docker environment using your own SSH credentials, ensuring secure access to the repository.

To clone the repository, use the following command in the terminal within the Docker container:

```bash
git clone git@github.com:chaitanya-m/auto_scientist.git

OR

```bash
chmod +x get_repo.sh
./get_repo.sh


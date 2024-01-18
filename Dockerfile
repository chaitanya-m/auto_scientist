# Start with the latest Ubuntu base image
FROM ubuntu:latest

# Update the package lists and install essential Python and development tools
# python3-pip: For installing Python packages
# python3-dev: Headers for Python development (required for some Python packages)
# build-essential: Essential tools for compiling and building software
# libssl-dev: SSL development libraries, needed for building Python packages that require secure connections
# libffi-dev: Foreign Function Interface library, needed by some Python packages
# python3-setuptools: Tools for building Python packages
# git: Version control system
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools git

# Set the working directory in the container to /workspace
# This is where the code will reside within the container
WORKDIR /workspace

# Copy everything from the current directory to the /workspace directory in the container
COPY . /workspace

# Add the Deadsnakes PPA to get access to newer Python versions
# software-properties-common: Allows you to manage additional software repositories
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

# Update package lists to include packages from Deadsnakes
RUN apt-get update

# Install Python 3.10 from Deadsnakes PPA
RUN apt-get install -y python3.10

# Set Python 3.10 as the default Python version
# This step ensures that "python" command invokes Python 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# If you have a requirements.txt file, you can uncomment these lines to install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Set the default command for the container to launch a bash shell
# This command runs when the container starts and gives you an interactive shell
CMD ["/bin/bash"]


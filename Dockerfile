# Start with an official Python base image which supports multiple architectures
FROM python:3.10-slim

# Install essential tools
RUN apt-get update && apt-get install -y wget git vim \
    build-essential gcc g++

# Set the working directory in the container to /workspace
WORKDIR /workspace

# Copy the script to get the repository into /workspace
COPY get_repo.sh /workspace/get_repo.sh

# Install Miniconda
RUN ARCH="$(uname -m)"; \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_VERSION="Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINICONDA_VERSION="Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $ARCH"; \
        exit 1; \
    fi; \
    wget https://repo.anaconda.com/miniconda/$MINICONDA_VERSION -O /miniconda.sh && \
    bash /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh

# Add Conda to PATH
ENV PATH=/miniconda/bin:$PATH

# Copy the environment.yml file into the container at /workspace
COPY environment.yml /workspace/

# Create the Conda environment
RUN conda env create -f environment.yml && rm /workspace/environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copy the startup script into /workspace
COPY startup.sh /workspace/startup.sh

# Set the scripts to be executable
RUN chmod +x /workspace/get_repo.sh /workspace/startup.sh

# Run the startup script when the container launches
CMD ["/workspace/startup.sh"]


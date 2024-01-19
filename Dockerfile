# Use an official base image
FROM ubuntu:latest

# Install wget and bzip2, which are required for installing Miniconda
RUN apt-get update && \
    apt-get install -y wget bzip2 git vim

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh

# Add Conda to PATH
ENV PATH=/miniconda/bin:$PATH

# Set the working directory in the container to /workspace
WORKDIR /workspace

# Copy the environment.yml file into the container at /workspace
COPY environment.yml /workspace/

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

########## VARIABLE BEHAVIOUR ##############

# Clone the repository at the end to ensure variability in build
RUN git clone git@github.com:chaitanya-m/auto_scientist.git /workspace


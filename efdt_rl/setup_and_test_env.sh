#!/bin/bash

# Define the name of your Conda environment.
ENV_NAME="myenv"
ENV_YML_PATH="environment.yml"

# Source Conda commands
source /miniconda/etc/profile.d/conda.sh

# Check if the Conda environment already exists.
if conda info --envs | grep -qw "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Updating it..."
    conda env update --name $ENV_NAME --file $ENV_YML_PATH --prune
else
    echo "Creating environment '$ENV_NAME'..."
    conda env create -f $ENV_YML_PATH
fi

echo "Environment setup completed."

# Activate the environment
conda activate $ENV_NAME

# Assuming test_sklearn.py is in the current directory. Adjust the path as necessary.
TEST_SCRIPT_PATH="./test_sklearn.py"
echo "Running test_sklearn.py within the '$ENV_NAME' environment..."
conda run -n $ENV_NAME python $TEST_SCRIPT_PATH

conda init
conda activate $ENV_NAME

